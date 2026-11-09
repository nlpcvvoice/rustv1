use super::*;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    time::timeout,
};

fn fields() -> Value {
    json!({
        "drain_request_id": sha(b"drain nonce"),
        "reason": "after_exact_recovery",
        "instance": "replacement-test",
        "pid": 123,
        "process_start_ticks": 456,
        "boot_id": "test-boot",
        "running_binary_sha256": sha(b"server"),
        "policy_sha256": sha(b"policy"),
        "launch_binding_sha256": sha(b"launch"),
        "recovery_evidence_sha256": sha(b"recovery"),
        "manifest_sha256": sha(b"manifest"),
        "quiesced_observation_sha256": sha(b"quiesced"),
        "preceding_ledger_sha256": sha(b"ledger"),
        "replacement_launch": "forbidden"
    })
}

fn exit_fields(request: &Value) -> Value {
    json!({
        "drain_request_id": request["fields"]["drain_request_id"],
        "request_seq": request["seq"],
        "instance": request["fields"]["instance"],
        "pid": request["fields"]["pid"],
        "termination_signal": "SIGTERM",
        "exit_status": 0,
        "exit_signal": null,
        "process_reaped": true,
        "replacement_launched": false
    })
}

#[test]
fn final_drain_rejects_wrong_target_nonzero_exit_and_relabelled_reap() {
    let request = json!({
        "kind": "orch_request_final_drain",
        "seq": 17,
        "fields": fields()
    });
    let exit = json!({
        "kind": "orch_final_exit",
        "fields": exit_fields(&request)
    });
    assert!(check_final_exit(&request, &exit).is_ok());

    for (field, changed) in [
        ("drain_request_id", json!(sha(b"other"))),
        ("request_seq", json!(0)),
        ("instance", json!("old-poisoned")),
        ("pid", json!(124)),
        ("termination_signal", json!("SIGKILL")),
        ("exit_status", json!(1)),
        ("exit_signal", json!(15)),
        ("process_reaped", json!(false)),
        ("replacement_launched", json!(true)),
        ("owner_thread_joined", json!(true)),
    ] {
        let mut bad = exit.clone();
        bad["fields"][field] = changed;
        assert!(check_final_exit(&request, &bad).is_err(), "{field}");
    }

    let mut bad = request.clone();
    bad["fields"]["replacement_launch"] = json!("permitted");
    assert!(check_final_exit(&bad, &exit).is_err());

    let mut bad = request.clone();
    bad["fields"]
        .as_object_mut()
        .unwrap()
        .remove("recovery_evidence_sha256");
    assert!(check_final_exit(&bad, &exit).is_err());

    let mut bad = exit.clone();
    bad["kind"] = json!("orch_restart_exit");
    assert!(check_final_exit(&request, &bad).is_err());
}

#[test]
fn final_drain_quiescence_preserves_frozen_lifecycle_and_execution_fields() {
    let before = json!({
        "instance": "replacement-test",
        "pid": 123,
        "lifecycle": "ready",
        "worker_alive": true,
        "ready": true,
        "forward_calls": 44,
        "startup_output_sha256": sha(b"startup")
    });
    let mut after = before.clone();
    after["ready"] = json!(false);
    assert!(quiescence(&before, &after).is_ok());

    for (field, changed) in [
        ("instance", json!("old")),
        ("pid", json!(124)),
        ("lifecycle", json!("stopped")),
        ("worker_alive", json!(false)),
        ("ready", json!(true)),
        ("forward_calls", json!(45)),
        ("startup_output_sha256", json!(sha(b"changed"))),
        ("owner_thread_joined", json!(true)),
    ] {
        let mut bad = after.clone();
        bad[field] = changed;
        assert!(quiescence(&before, &bad).is_err(), "{field}");
    }

    let mut poisoned = before.clone();
    poisoned["lifecycle"] = json!("poisoned");
    assert!(quiescence(&poisoned, &after).is_err());
}

#[test]
fn final_drain_saved_interval_rejects_rebaseline_gap_and_byte_rewrite() {
    let root = std::env::temp_dir().join(format!(
        "rv1-final-drain-interval-{}-{}",
        std::process::id(),
        core::random_label().unwrap()
    ));
    private_directory(&root).unwrap();

    let result: Result<()> = (|| {
        let prefix = b"{ \"startup\": true }\n";
        let suffix = b"{\"begin\":1}\n{\"end\":1}\n";
        let mut manifest = prefix.to_vec();
        manifest.extend_from_slice(suffix);
        write_new(&root.join("interval.jsonl"), suffix)?;

        let proof = json!({
            "manifest_interval": {
                "path": "interval.jsonl",
                "start_byte": prefix.len(),
                "end_byte": manifest.len(),
                "prefix_before_sha256": sha(prefix),
                "prefix_after_sha256": sha(&manifest),
                "interval_sha256": sha(suffix)
            }
        });
        ensure!(
            saved_interval(&root, &proof, &manifest, prefix.len())? == suffix,
            "valid saved interval"
        );
        ensure!(
            saved_interval(&root, &proof, &manifest, prefix.len() - 1).is_err(),
            "accepted interval gap"
        );

        let mut bad = proof.clone();
        bad["manifest_interval"]["interval_sha256"] = json!(sha(b"new baseline"));
        ensure!(
            saved_interval(&root, &bad, &manifest, prefix.len()).is_err(),
            "accepted changed commitment"
        );

        let mut changed = manifest.clone();
        changed[1] = b'\t';
        ensure!(
            saved_interval(&root, &proof, &changed, prefix.len()).is_err(),
            "accepted preceding-byte rewrite"
        );

        fs::write(
            &root.join("interval.jsonl"),
            b"{\"begin\":2}\n{\"end\":1}\n",
        )?;
        ensure!(
            saved_interval(&root, &proof, &manifest, prefix.len()).is_err(),
            "accepted saved interval mutation"
        );
        Ok(())
    })();

    if result.is_ok() {
        fs::remove_dir_all(&root).unwrap();
    }
    result
        .unwrap_or_else(|error| panic!("interval harness failed at {}: {error:#}", root.display()));
}

async fn read_wire(stream: &mut TcpStream) -> Result<Value> {
    let mut bytes = Vec::new();
    loop {
        let byte = stream.read_u8().await?;
        if byte == b'\n' {
            break;
        }
        ensure!(bytes.len() < 1024 * 1024, "test frame bound");
        bytes.push(byte);
    }
    Ok(serde_json::from_slice(&bytes)?)
}

async fn send_wire(
    stream: &mut TcpStream,
    key: &[u8],
    kind: &str,
    seq: u64,
    fields: Value,
) -> Result<()> {
    let mut envelope = json!({"kind": kind, "seq": seq, "fields": fields});
    let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, key);
    let tag = ring::hmac::sign(&key, core::canonical_json(&envelope)?.as_bytes());
    envelope["mac"] = json!(tag
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>());
    let mut bytes = serde_json::to_vec(&envelope)?;
    bytes.push(b'\n');
    stream.write_all(&bytes).await?;
    Ok(())
}

fn verify_wire(envelope: &Value, key: &[u8], seq: u64) -> Result<()> {
    closed(envelope, &["kind", "seq", "fields", "mac"])?;
    ensure!(envelope["seq"] == seq, "test directional sequence");
    let payload = json!({
        "kind": envelope["kind"],
        "seq": envelope["seq"],
        "fields": envelope["fields"]
    });
    let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, key);
    ring::hmac::verify(
        &key,
        core::canonical_json(&payload)?.as_bytes(),
        &decode_hex(text(envelope, "mac")?)?,
    )
    .map_err(|_| anyhow::anyhow!("test request MAC"))?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn final_drain_authenticated_exchange_preserves_sequences_and_rejects_replay() {
    let root = std::env::temp_dir().join(format!(
        "rv1-final-drain-wire-{}-{}",
        std::process::id(),
        core::random_label().unwrap()
    ));
    private_directory(&root).unwrap();

    let result: Result<()> = async {
        for replay in [false, true] {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let address = listener.local_addr()?;
            let send_key = [0x31u8; 32];
            let receive_key = [0x72u8; 32];

            let peer = tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await?;
                let seed = read_wire(&mut stream).await?;
                verify_wire(&seed, &send_key, 0)?;
                ensure!(seed["kind"] == "test_seed", "seed kind");
                send_wire(&mut stream, &receive_key, "test_seed_ack", 0, json!({})).await?;

                let request = read_wire(&mut stream).await?;
                verify_wire(&request, &send_key, 1)?;
                ensure!(
                    request["kind"] == "orch_request_final_drain",
                    "final request kind"
                );
                validate_drain_fields(&request["fields"])?;
                send_wire(
                    &mut stream,
                    &receive_key,
                    "orch_final_exit",
                    if replay { 0 } else { 1 },
                    exit_fields(&request),
                )
                .await?;
                Ok::<(), anyhow::Error>(())
            });

            let client_work: Result<()> = async {
                let stream = TcpStream::connect(address).await?;
                let path = root.join(if replay {
                    "replay.jsonl"
                } else {
                    "valid.jsonl"
                });
                let mut orchestration = core::Orchestration::from_test_stream(
                    stream,
                    &send_key,
                    &receive_key,
                    Duration::from_secs(5),
                    &path,
                )?;
                orchestration.send("test_seed", json!({})).await?;
                ensure!(
                    orchestration.receive().await?["kind"] == "test_seed_ack",
                    "seed acknowledgement"
                );

                let exchange = drain_exchange(&mut orchestration, fields()).await;
                if replay {
                    ensure!(exchange.is_err(), "accepted reset receive sequence");
                    ensure!(
                        orchestration
                            .send("test_after_failure", json!({}))
                            .await
                            .is_err(),
                        "failed orchestration connection remained usable"
                    );
                } else {
                    let (request, exit) = exchange?;
                    ensure!(
                        request["seq"] == 1 && exit["seq"] == 1,
                        "final drain reset directional counters"
                    );
                    let policy = json!({
                        "orch_secret_server_to_orchestrator": "31".repeat(32),
                        "orch_secret_orchestrator_to_server": "72".repeat(32)
                    });
                    let records = verify_transcript(&bounded(&path, JSON_LIMIT)?, &policy)?;
                    ensure!(
                        records.len() == 4
                            && records[2]["envelope"] == request
                            && records[3]["envelope"] == exit,
                        "authenticated transcript suffix"
                    );
                }
                Ok(())
            }
            .await;

            // Always collect the test supervisor task before propagating errors.
            let peer_work = timeout(Duration::from_secs(10), peer)
                .await
                .context("test supervisor watchdog")?
                .context("test supervisor task")?;
            client_work?;
            peer_work?;
        }
        Ok(())
    }
    .await;

    if result.is_ok() {
        fs::remove_dir_all(&root).unwrap();
    }
    result.unwrap_or_else(|error| panic!("wire harness failed at {}: {error:#}", root.display()));
}
