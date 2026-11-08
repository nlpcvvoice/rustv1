use super::*;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

fn fields() -> Value {
    json!({
        "launch_request_id": sha(b"recovery nonce"),
        "reason": "after_poison_exit",
        "restart_request_id": sha(b"restart nonce"),
        "restart_request_seq": 7,
        "previous_instance": "poisoned",
        "previous_pid": 42,
        "previous_process_start_ticks": 123,
        "previous_boot_id": "test-boot",
        "policy_sha256": sha(b"policy"),
        "previous_launch_binding_sha256": sha(b"old binding"),
        "exit_evidence_sha256": sha(b"saved exit"),
        "old_absence_sha256": sha(b"independent absence")
    })
}

fn accepted(request: &Value) -> Value {
    json!({
        "kind": "orch_recovery_launch_accepted",
        "seq": 0,
        "fields": {
            "launch_request_id": request["fields"]["launch_request_id"],
            "request_seq": request["seq"],
            "restart_request_id": request["fields"]["restart_request_id"],
            "previous_instance": request["fields"]["previous_instance"],
            "previous_pid": request["fields"]["previous_pid"]
        }
    })
}

fn ready(request: &Value) -> Value {
    json!({
        "kind": "orch_ready",
        "seq": 1,
        "fields": {
            "launch_request_id": request["fields"]["launch_request_id"],
            "instance": "fresh-recovery",
            "pid": 43,
            "start_utc": "123456.000001Z",
            "manifest": "/test/recovery-manifest.jsonl",
            "attestation": "/test/recovery-attestation.json"
        }
    })
}

#[test]
fn recovery_launch_rejects_stale_extra_and_reused_identity() {
    let request = json!({
        "kind": "orch_request_recovery_launch",
        "seq": 8,
        "fields": fields()
    });
    let ack = accepted(&request);
    let ready_frame = ready(&request);
    assert!(check_launch_accepted(&request, &ack).is_ok());
    assert!(check_ready(&request, &ready_frame).is_ok());

    for (name, changed) in [
        ("launch_request_id", json!(sha(b"stale"))),
        ("request_seq", json!(7)),
        ("restart_request_id", json!(sha(b"other restart"))),
        ("previous_instance", json!("different")),
        ("previous_pid", json!(44)),
        ("extra", json!(true)),
    ] {
        let mut bad = ack.clone();
        bad["fields"][name] = changed;
        assert!(check_launch_accepted(&request, &bad).is_err(), "{name}");
    }

    for (name, changed) in [
        ("launch_request_id", json!(sha(b"stale"))),
        ("instance", json!("poisoned")),
        ("pid", json!(42)),
        ("manifest", json!("relative.jsonl")),
        ("attestation", json!("relative.json")),
        ("owner_thread_joined", json!(true)),
    ] {
        let mut bad = ready_frame.clone();
        bad["fields"][name] = changed;
        assert!(check_ready(&request, &bad).is_err(), "{name}");
    }

    let mut wrong_kind = ack;
    wrong_kind["kind"] = json!("orch_launch_accepted");
    assert!(check_launch_accepted(&request, &wrong_kind).is_err());

    let mut premature = ready_frame;
    premature["kind"] = json!("orch_restart_exit");
    assert!(check_ready(&request, &premature).is_err());
}

#[test]
fn recovery_preserves_historical_poison_and_exact_byte_prefixes() {
    let observation = json!({
        "instance": "poisoned",
        "pid": 42,
        "lifecycle": "poisoned",
        "worker_alive": true,
        "ready": false,
        "forward_calls": 9,
        "startup_output_sha256": sha(b"startup")
    });
    let manifest = b"{\"kind\":\"saved\"}\n";
    let ids = BTreeSet::from([1, 3, 7]);
    let exit = json!({
        "old_pre_exit_observation": observation,
        "old_observation_is_historical": true,
        "manifest_unchanged_sha256": sha(manifest),
        "executor_next_forward_id_unchanged": 8,
        "exposed_request_ids_unchanged": ids,
        "service_owner_join_claimed": false,
        "quarantine_cleared_in_process": false,
        "replacement_launch_authorized": false
    });
    assert!(historical_cursor(&exit, &observation, manifest, 8, &ids).is_ok());

    for (name, changed) in [
        ("service_owner_join_claimed", json!(true)),
        ("quarantine_cleared_in_process", json!(true)),
        ("executor_next_forward_id_unchanged", json!(9)),
        ("exposed_request_ids_unchanged", json!([1, 3])),
    ] {
        let mut bad = exit.clone();
        bad[name] = changed;
        assert!(historical_cursor(&bad, &observation, manifest, 8, &ids).is_err());
    }
    let mut relabelled = observation.clone();
    relabelled["lifecycle"] = json!("stopped");
    assert!(historical_cursor(&exit, &relabelled, manifest, 8, &ids).is_err());

    let bytes = b"{ \"a\": 1 }\n{\"b\":2}\n";
    assert_eq!(prefix_before(bytes, 0).unwrap(), b"");
    assert_eq!(prefix_before(bytes, 1).unwrap(), b"{ \"a\": 1 }\n");
    assert_eq!(prefix_before(bytes, 2).unwrap(), bytes);
    assert!(prefix_before(bytes, 3).is_err());
    assert!(prefix_before(b"{}\n{}", 1).is_err());
}

#[test]
fn recovery_bound_artifacts_reject_mutation_and_hardlinks() {
    let root = std::env::temp_dir().join(format!(
        "rv1-recovery-artifacts-{}-{}",
        std::process::id(),
        core::random_label().unwrap()
    ));
    private_directory(&root).unwrap();

    let result: Result<()> = (|| {
        let original = b"{\"old\":\"evidence\"}";
        write_new(&root.join("original.json"), original)?;
        let digest = sha(original);
        ensure!(
            read_bound(&root, "original.json", &digest)? == original,
            "bound read"
        );

        fs::hard_link(root.join("original.json"), root.join("alias.json"))?;
        ensure!(
            distinct_files(&root.join("original.json"), &root.join("alias.json")).is_err(),
            "hard-linked recovery evidence accepted"
        );

        fs::write(root.join("original.json"), b"{\"old\":\"changed\"}")?;
        ensure!(
            read_bound(&root, "original.json", &digest).is_err(),
            "mutated artifact was rebaselined"
        );
        ensure!(
            read_bound(&root, "../outside.json", &digest).is_err(),
            "traversal accepted"
        );
        Ok(())
    })();

    if result.is_ok() {
        fs::remove_dir_all(&root).unwrap();
    }
    result
        .unwrap_or_else(|error| panic!("artifact harness failed at {}: {error:#}", root.display()));
}

async fn read_frame(socket: &mut TcpStream) -> Result<Value> {
    let mut bytes = Vec::new();
    loop {
        let byte = socket.read_u8().await?;
        if byte == b'\n' {
            ensure!(!bytes.is_empty(), "empty supervisor frame");
            return Ok(serde_json::from_slice(&bytes)?);
        }
        ensure!(bytes.len() < 1024 * 1024, "supervisor frame bound");
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

async fn send_frame(socket: &mut TcpStream, envelope: Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(&envelope)?;
    bytes.push(b'\n');
    socket.write_all(&bytes).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recovery_transport_authenticates_exact_exchange_and_rejects_bad_mac() {
    let root = std::env::temp_dir().join(format!(
        "rv1-recovery-channel-{}-{}",
        std::process::id(),
        core::random_label().unwrap()
    ));
    private_directory(&root).unwrap();

    let result: Result<()> = async {
        let send_key = [0x31u8; 32];
        let receive_key = [0x42u8; 32];
        let policy_value = json!({
            "orch_secret_server_to_orchestrator": "31".repeat(32),
            "orch_secret_orchestrator_to_server": "42".repeat(32)
        });

        for bad_mac in [false, true] {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let address = listener.local_addr()?;
            let peer_policy = policy_value.clone();

            let peer = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await?;
                let request = read_frame(&mut socket).await?;
                ensure!(
                    request["kind"] == "orch_request_recovery_launch"
                        && request["seq"] == 0
                        && request["fields"] == fields(),
                    "wrong recovery authorization"
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
                send_frame(&mut socket, ack).await?;
                if !bad_mac {
                    send_frame(&mut socket, sign(ready(&request), &receive_key)?).await?;
                }
                Ok::<(), anyhow::Error>(())
            });

            let path = root.join(if bad_mac { "bad.jsonl" } else { "good.jsonl" });
            let stream = TcpStream::connect(address).await?;
            let mut orchestration = core::Orchestration::from_test_stream(
                stream,
                &send_key,
                &receive_key,
                Duration::from_secs(5),
                &path,
            )?;

            let exchange = tokio::time::timeout(
                Duration::from_secs(10),
                authorize(&mut orchestration, fields()),
            )
            .await
            .context("recovery protocol harness watchdog");

            // Await the peer on both success and rejection paths.
            let peer_result = peer.await.context("supervisor task join")?;
            peer_result?;
            let exchange = exchange?;

            if bad_mac {
                ensure!(exchange.is_err(), "unauthenticated acceptance accepted");
                ensure!(
                    orchestration
                        .send("must-not-send", json!({}))
                        .await
                        .is_err(),
                    "bad MAC did not make transport terminal"
                );
                ensure!(
                    verify_transcript(&bounded(&path, JSON_LIMIT)?, &policy_value)?.len() == 1,
                    "unauthenticated supervisor frame was recorded"
                );
            } else {
                let (request, accepted, ready) = exchange?;
                orchestration.sync()?;
                let saved = verify_transcript(&bounded(&path, JSON_LIMIT)?, &policy_value)?;
                ensure!(
                    saved
                        == vec![
                            json!({
                                "direction": "driver_to_orchestrator",
                                "envelope": request
                            }),
                            json!({
                                "direction": "orchestrator_to_driver",
                                "envelope": accepted
                            }),
                            json!({
                                "direction": "orchestrator_to_driver",
                                "envelope": ready
                            }),
                        ],
                    "actual transport exchange differs from expected transcript"
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
            "recovery channel harness failed; evidence retained at {}: {error:#}",
            root.display()
        )
    });
}
