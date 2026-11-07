use super::*;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

fn fields() -> Value {
    json!({
        "restart_request_id": sha(b"restart nonce"),
        "reason": "poison_terminal",
        "instance": "old-instance",
        "pid": 42,
        "process_start_ticks": 1234,
        "boot_id": "test-boot",
        "running_binary_sha256": sha(b"server"),
        "policy_sha256": sha(b"policy"),
        "launch_binding_sha256": sha(b"launch"),
        "restore_evidence_sha256": sha(b"saved restore"),
        "manifest_sha256": sha(b"saved manifest"),
        "replacement_launch": LAUNCH_GATE
    })
}

fn accepted(request: &Value) -> Value {
    json!({
        "kind": "orch_restart_accepted",
        "seq": 0,
        "fields": {
            "restart_request_id": request["fields"]["restart_request_id"],
            "request_seq": request["seq"],
            "instance": request["fields"]["instance"],
            "pid": request["fields"]["pid"],
            "replacement_launch": LAUNCH_GATE
        }
    })
}

fn exited(request: &Value) -> Value {
    json!({
        "kind": "orch_restart_exit",
        "seq": 1,
        "fields": {
            "restart_request_id": request["fields"]["restart_request_id"],
            "request_seq": request["seq"],
            "instance": request["fields"]["instance"],
            "pid": request["fields"]["pid"],
            "termination_signal": "SIGTERM",
            "exit_status": 1,
            "exit_signal": null,
            "process_reaped": true,
            "replacement_launched": false
        }
    })
}

#[test]
fn restart_protocol_rejects_stale_extra_and_relabelled_exit_fields() {
    let request = json!({
        "kind": "orch_request_restart",
        "seq": 9,
        "fields": fields()
    });
    let ack = accepted(&request);
    assert!(check_accepted(&request, &ack).is_ok());

    for (key, changed) in [
        ("request_seq", json!(8)),
        ("restart_request_id", json!(sha(b"other nonce"))),
        ("pid", json!(43)),
        ("replacement_launch", json!("immediate")),
        ("unexpected", json!(true)),
    ] {
        let mut bad = ack.clone();
        bad["fields"][key] = changed;
        assert!(check_accepted(&request, &bad).is_err(), "{key}");
    }

    let exit = exited(&request);
    assert!(check_exit(&request, &exit).is_ok());
    for (key, changed) in [
        ("request_seq", json!(8)),
        ("instance", json!("other-instance")),
        ("exit_status", json!(0)),
        ("exit_signal", json!(9)),
        ("termination_signal", json!("SIGKILL")),
        ("process_reaped", json!(false)),
        ("replacement_launched", json!(true)),
        ("owner_thread_joined", json!(true)),
    ] {
        let mut bad = exit.clone();
        bad["fields"][key] = changed;
        assert!(check_exit(&request, &bad).is_err(), "{key}");
    }

    let mut premature_ready = exit;
    premature_ready["kind"] = json!("orch_ready");
    assert!(check_exit(&request, &premature_ready).is_err());
}

async fn read_frame(stream: &mut TcpStream) -> Result<Value> {
    let mut bytes = Vec::new();
    loop {
        let byte = stream.read_u8().await?;
        if byte == b'\n' {
            ensure!(!bytes.is_empty(), "empty mock-supervisor frame");
            return Ok(serde_json::from_slice(&bytes)?);
        }
        ensure!(bytes.len() < 1024 * 1024, "mock-supervisor frame bound");
        bytes.push(byte);
    }
}

fn sign(mut envelope: Value, key: &[u8]) -> Result<Value> {
    let payload = core::canonical_json(&envelope)?;
    let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, key);
    let tag = ring::hmac::sign(&key, payload.as_bytes());
    envelope["mac"] = json!(tag
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>());
    Ok(envelope)
}

async fn send_frame(stream: &mut TcpStream, envelope: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(envelope)?;
    bytes.push(b'\n');
    stream.write_all(&bytes).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn authenticated_restart_channel_binds_request_and_rejects_bad_mac() {
    let root = std::env::temp_dir().join(format!(
        "rv1-restart-protocol-{}-{}",
        std::process::id(),
        core::random_label().unwrap()
    ));
    private_directory(&root).unwrap();

    let result: Result<()> = async {
        let send_key = [0x11u8; 32];
        let receive_key = [0x22u8; 32];
        let policy_value = json!({
            "orch_secret_server_to_orchestrator": "11".repeat(32),
            "orch_secret_orchestrator_to_server": "22".repeat(32)
        });

        for bad_mac in [false, true] {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let address = listener.local_addr()?;
            let expected_fields = fields();
            let peer_policy = policy_value.clone();

            let peer = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await?;
                let request = read_frame(&mut socket).await?;
                ensure!(
                    request["kind"] == "orch_request_restart"
                        && request["seq"] == 0
                        && request["fields"] == expected_fields,
                    "mock supervisor received wrong request"
                );
                let mut record = serde_json::to_vec(&json!({
                    "direction": "driver_to_orchestrator",
                    "envelope": request
                }))?;
                record.push(b'\n');
                verify_transcript(&record, &peer_policy)?;

                let mut ack = sign(accepted(&request), &receive_key)?;
                if bad_mac {
                    ack["mac"] = json!("00".repeat(32));
                }
                send_frame(&mut socket, &ack).await?;
                if !bad_mac {
                    let exit = sign(exited(&request), &receive_key)?;
                    send_frame(&mut socket, &exit).await?;
                }
                Ok::<(), anyhow::Error>(())
            });

            let path = root.join(if bad_mac {
                "bad-mac.jsonl"
            } else {
                "valid.jsonl"
            });
            let stream = TcpStream::connect(address).await?;
            let mut orchestration = core::Orchestration::from_test_stream(
                stream,
                &send_key,
                &receive_key,
                Duration::from_secs(5),
                &path,
            )?;

            let exchange = request_restart(&mut orchestration, fields()).await;
            // Always await the mock supervisor, including the rejection path.
            let peer_result = peer.await.context("mock supervisor task")?;
            peer_result?;

            if bad_mac {
                ensure!(exchange.is_err(), "bad MAC accepted");
                ensure!(
                    orchestration
                        .send("must-not-send", json!({}))
                        .await
                        .is_err(),
                    "failed authentication did not make channel terminal"
                );
                let saved = verify_transcript(&bounded(&path, JSON_LIMIT)?, &policy_value)?;
                ensure!(saved.len() == 1, "unauthenticated frame was recorded");
            } else {
                let (request, ack) = exchange?;
                let exit = receive_exit(&mut orchestration, &request).await?;
                check_accepted(&request, &ack)?;
                check_exit(&request, &exit)?;
                orchestration.sync()?;

                let saved = verify_transcript(&bounded(&path, JSON_LIMIT)?, &policy_value)?;
                ensure!(
                    saved.len() == 3
                        && saved[0]["envelope"] == request
                        && saved[1]["envelope"] == ack
                        && saved[2]["envelope"] == exit,
                    "actual transport transcript binding"
                );
            }
        }
        Ok(())
    }
    .await;

    if result.is_ok() {
        fs::remove_dir_all(&root).unwrap();
    }
    result.unwrap_or_else(|error| {
        panic!(
            "restart protocol harness failed; evidence retained at {}: {error:#}",
            root.display()
        )
    });
}
