// Included at acceptance_http module scope, only with scheduler-test-seam.

mod connection_owners {
    use super::*;
    use std::{
        future::{poll_fn, Future},
        sync::{Mutex, MutexGuard, OnceLock},
    };

    type ConnectionResult = std::result::Result<(), String>;

    struct Entry {
        handle: Option<JoinHandle<ConnectionResult>>,
        identity: Value,
        abort_requested: bool,
        outcome: Option<Value>,
    }

    #[derive(Default)]
    struct Inventory {
        next: u64,
        entries: BTreeMap<u64, Entry>,
    }

    fn inventory() -> &'static Mutex<Inventory> {
        static INVENTORY: OnceLock<Mutex<Inventory>> = OnceLock::new();
        INVENTORY.get_or_init(|| Mutex::new(Inventory::default()))
    }

    fn locked() -> MutexGuard<'static, Inventory> {
        inventory().lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(super) struct Owner {
        id: u64,
        disarmed: bool,
    }

    pub(super) fn retain(
        handle: JoinHandle<ConnectionResult>,
        mut identity: Value,
    ) -> Owner {
        let mut inventory = locked();
        let id = inventory.next;
        inventory.next = id.checked_add(1).expect("connection inventory overflow");
        identity["connection_id"] = json!(id);
        identity["owner_pid"] = json!(std::process::id());
        identity["tokio_task_id"] = json!(handle.id().to_string());
        assert!(inventory.entries.insert(id, Entry {
            handle: Some(handle),
            identity,
            abort_requested: false,
            outcome: None,
        }).is_none());
        Owner { id, disarmed: false }
    }

    fn abort(id: u64) {
        let mut inventory = locked();
        let entry = inventory.entries.get_mut(&id).expect("retained connection");
        if let Some(handle) = &entry.handle {
            entry.abort_requested = true;
            handle.abort();
        }
    }

    // Cancellation-safe: the handle always stays in Inventory. Once its poll
    // returns Ready, consumption and evidence publication have no await gap.
    async fn joined(id: u64) -> Value {
        poll_fn(|cx| {
            let mut inventory = locked();
            let entry = inventory.entries.get_mut(&id).expect("retained connection");
            if let Some(outcome) = &entry.outcome {
                return Poll::Ready(outcome.clone());
            }

            let handle = entry.handle.as_mut().expect("unjoined connection handle");
            let result = match Pin::new(handle).poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(result) => result,
            };
            entry.handle.take();

            let (outcome, connection_result, join_error) = match result {
                Ok(result) => (
                    "completed",
                    json!({
                        "ok": result.is_ok(),
                        "error": result.err()
                    }),
                    Value::Null,
                ),
                Err(error) if error.is_cancelled() => {
                    // Only this inventory can abort these handles. Merely
                    // observing is_finished() is never used as a join.
                    assert!(entry.abort_requested);
                    (
                        "cancelled-after-abort-and-awaited",
                        Value::Null,
                        json!(error.to_string()),
                    )
                }
                Err(error) => (
                    "panicked",
                    Value::Null,
                    json!(error.to_string()),
                ),
            };

            let mut record = entry.identity.clone();
            record["join_outcome"] = json!(outcome);
            record["abort_requested"] = json!(entry.abort_requested);
            record["connection_result"] = connection_result;
            record["join_error"] = join_error;
            record["join_awaited"] = json!(true);
            entry.outcome = Some(record.clone());
            Poll::Ready(record)
        }).await
    }

    impl Owner {
        pub(super) async fn finish(
            &mut self,
            ordinary_error: bool,
            budget: Duration,
        ) -> Value {
            if ordinary_error {
                abort(self.id);
            }

            let record = match timeout(budget, joined(self.id)).await {
                Ok(record) => record,
                Err(_) => {
                    abort(self.id);
                    joined(self.id).await
                }
            };
            self.disarmed = true;
            record
        }
    }

    impl Drop for Owner {
        fn drop(&mut self) {
            if !self.disarmed {
                // No detached reaper task and no dropped JoinHandle.
                abort(self.id);
            }
        }
    }

    pub(super) async fn close_instance(instance: &str) -> Vec<Value> {
        let ids: Vec<u64> = {
            let inventory = locked();
            inventory.entries.iter()
                .filter(|(_, entry)| entry.identity["instance"] == instance)
                .map(|(id, _)| *id)
                .collect()
        };

        let mut records = Vec::with_capacity(ids.len());
        for id in ids {
            // The caller closes an instance only after its request futures
            // have completed or have themselves been aborted and awaited.
            abort(id);
            records.push(joined(id).await);
        }
        records
    }
}

pub(crate) async fn join_connections(instance: &str) -> Result<Vec<Value>> {
    Ok(connection_owners::close_instance(instance).await)
}

impl Http {
    pub(crate) async fn send(&self, case: &str, wire: &Wire) -> Result<Reply> {
        self.send_joined(case, "POST", "/v1/embeddings", Some(wire)).await
    }

    pub(crate) async fn get_health(&self, case: &str, path: &str) -> Result<Reply> {
        ensure!(
            matches!(path, "/health/livez" | "/health/readyz"),
            "unsupported health route"
        );
        self.send_joined(case, "GET", path, None).await
    }

    async fn send_joined(
        &self,
        case: &str,
        method: &str,
        path: &str,
        wire: Option<&Wire>,
    ) -> Result<Reply> {
        ensure!(label_valid(case), "invalid case label");
        let request_bytes = wire.map_or(&[][..], |wire| wire.body.as_slice());
        ensure!(request_bytes.len() <= BODY_LIMIT + 65536, "request size bound");

        let stream = timeout(
            Duration::from_millis(self.limits.connect_ms),
            TcpStream::connect(self.address),
        ).await.context("HTTP connect deadline")??;
        let local_address = stream.local_addr()?;
        let peer_address = stream.peer_addr()?;

        let mut builder = http1::Builder::new();
        builder.max_headers(64);
        builder.max_buf_size(64 * 1024);
        let (mut sender, connection) = timeout(
            Duration::from_millis(self.limits.connect_ms),
            builder.handshake(TokioIo::new(stream)),
        ).await.context("HTTP handshake deadline")??;

        // No cancellation point between spawning and retaining this handle.
        let mut owner = connection_owners::retain(
            tokio::spawn(async move {
                connection.await.map_err(|error| error.to_string())
            }),
            json!({
                "instance": self.instance,
                "case_id": case,
                "method": method,
                "path": path,
                "local_address": local_address.to_string(),
                "peer_address": peer_address.to_string(),
                "request_sha256": sha(request_bytes)
            }),
        );

        // Every ordinary error after connection creation exits this block,
        // drops the response/body, drops sender, then awaits the connection.
        let result: Result<Reply> = async {
            let mut request = Request::builder()
                .method(method)
                .uri(path)
                .header("host", &self.authority)
                .header("accept", "application/json")
                .header("accept-encoding", "identity")
                .header("connection", "close")
                .header("x-acceptance-case-id", case);
            if let Some(wire) = wire {
                request = request.header("content-type", &wire.media);
                if let Some(encoding) = &wire.content_encoding {
                    request = request.header("content-encoding", encoding);
                }
            }
            let request = request.body(RequestBody {
                bytes: Bytes::copy_from_slice(request_bytes),
                chunked: wire.is_some_and(|wire| wire.chunked),
            })?;
            let response = timeout(
                Duration::from_millis(self.limits.headers_ms),
                sender.send_request(request),
            ).await.context("HTTP response-header deadline")??;

            let headers = response.headers();
            ensure!(
                one_header(headers, "x-rv1-instance-id")? == Some(self.instance.as_str())
                    && one_header(headers, "x-acceptance-case-id")? == Some(case),
                "HTTP correlation headers"
            );
            let status = response.status().as_u16();
            let media = one_header(headers, "content-type")?;
            if method == "POST" || status != 200 {
                ensure!(
                    media.is_some_and(|media| media.split(';').next()
                        .unwrap_or("").trim().eq_ignore_ascii_case("application/json")),
                    "HTTP response media"
                );
            }
            ensure!(
                one_header(headers, "content-encoding")?.is_none(),
                "encoded response forbidden"
            );

            let request_id = one_header(headers, "x-rv1-request-id")?
                .map(|text| -> Result<u64> {
                    let id: u64 = text.parse()?;
                    ensure!(id > 0 && text == id.to_string(), "noncanonical request ID");
                    Ok(id)
                }).transpose()?;

            let recorded_headers = json!({
                "x-rv1-instance-id": self.instance,
                "x-acceptance-case-id": case,
                "x-rv1-request-id": request_id,
                "content-type": media
            });

            let mut body = response.into_body();
            let bytes = timeout(Duration::from_millis(self.limits.body_ms), async {
                let mut bytes = Vec::new();
                while let Some(frame) = body.frame().await {
                    let frame = frame?;
                    let data = frame.data_ref().context("unexpected response trailers")?;
                    ensure!(
                        data.len() <= self.limits.response_bytes.saturating_sub(bytes.len()),
                        "response body exceeds bound"
                    );
                    bytes.extend_from_slice(data);
                }
                Ok::<_, anyhow::Error>(bytes)
            }).await.context("HTTP response-body deadline")??;

            Ok(Reply { status, request_id, headers: recorded_headers, bytes })
        }.await;

        drop(sender);
        let joined = owner.finish(
            result.is_err(),
            Duration::from_millis(self.limits.body_ms),
        ).await;

        // Each successful return carries the exact registry record. Existing
        // scheduler response evidence already saves Reply.headers, so this
        // record becomes part of the preceding digest-bound evidence.
        let mut reply = result?;
        ensure!(joined["join_outcome"] != "panicked", "HTTP connection task panicked");
        reply.headers["_connection_owner"] = joined;
        Ok(reply)
    }
}

#[cfg(test)]
mod joined_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn strict_connection_ownership_outcomes_and_request_cancellation() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rv1-strict-joins-{}-{stamp}", std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let instance = format!("joins-{}-{stamp}", std::process::id());

        let result: Result<()> = async {
            for (case, result) in [
                ("normal", Ok(())),
                ("transport-error", Err("measured transport error".to_owned())),
            ] {
                let mut owner = connection_owners::retain(
                    tokio::spawn(async move { result }),
                    json!({"instance": instance, "case_id": case}),
                );
                let record = owner.finish(false, Duration::from_secs(5)).await;
                ensure!(record["join_outcome"] == "completed", "completed join classification");
            }

            let mut panicked = connection_owners::retain(
                tokio::spawn(async {
                    panic!("intentional connection-owner test panic");
                    #[allow(unreachable_code)]
                    Ok(())
                }),
                json!({"instance": instance, "case_id": "panic"}),
            );
            let record = panicked.finish(false, Duration::from_secs(5)).await;
            ensure!(record["join_outcome"] == "panicked", "panic classification");

            let owner = connection_owners::retain(
                tokio::spawn(async {
                    std::future::pending::<()>().await;
                    Ok(())
                }),
                json!({"instance": instance, "case_id": "cancelled-request"}),
            );
            let (entered, entry) = tokio::sync::oneshot::channel();
            let request = tokio::spawn(async move {
                let _owner = owner;
                let _ = entered.send(());
                std::future::pending::<()>().await;
            });
            entry.await?;
            request.abort();
            ensure!(request.await.unwrap_err().is_cancelled(), "request task cancellation");

            let records = join_connections(&instance).await?;
            write_new(
                &root.join("connection_owners.json"),
                &serde_json::to_vec_pretty(&records)?,
            )?;
            ensure!(records.len() == 4, "owner coverage");
            let cancelled = records.iter()
                .find(|r| r["case_id"] == "cancelled-request").context("cancelled owner")?;
            ensure!(
                cancelled["join_outcome"] == "cancelled-after-abort-and-awaited"
                    && cancelled["abort_requested"] == true
                    && cancelled["join_awaited"] == true,
                "cancelled request lost its connection ownership"
            );
            let transport = records.iter()
                .find(|r| r["case_id"] == "transport-error").context("transport owner")?;
            ensure!(
                transport["connection_result"]["ok"] == false
                    && transport["join_outcome"] == "completed",
                "transport error confused with task panic/cancellation"
            );
            Ok(())
        }.await;

        if result.is_ok() {
            fs::remove_dir_all(&root).unwrap();
        }
        result.unwrap_or_else(|error| panic!(
            "strict connection harness failed; evidence retained at {}: {error:#}",
            root.display()
        ));
    }
}
