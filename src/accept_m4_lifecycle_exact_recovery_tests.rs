use super::*;

fn row() -> Vec<f32> {
    let mut values = vec![0.0; crate::contract::WIDTH];
    values[0] = 1.0;
    values
}

#[test]
fn exact_recovery_rejects_tolerance_only_signed_zero_and_wrong_baseline() {
    let frozen = row();
    let digest = sha(&row_bytes(&frozen));
    assert_eq!(
        exact_singleton(&[frozen.clone()], &frozen, &digest).unwrap(),
        digest
    );

    // This passes the numerical tolerance gate but must fail exact recovery.
    let mut close = frozen.clone();
    close[1] = 0.000001;
    assert!(compare(&close, &frozen).is_ok());
    assert!(exact_singleton(&[close], &frozen, &digest).is_err());

    // IEEE signed zero is numerically equal, but not exact F32 bits.
    let mut signed_zero = frozen.clone();
    signed_zero[1] = -0.0;
    assert!(compare(&signed_zero, &frozen).is_ok());
    assert!(exact_singleton(&[signed_zero], &frozen, &digest).is_err());

    assert!(exact_singleton(&[], &frozen, &digest).is_err());
    assert!(exact_singleton(&[frozen.clone(), frozen.clone()], &frozen, &digest,).is_err());
    assert!(exact_singleton(&[frozen.clone()], &frozen, &sha(b"other")).is_err());

    let mut nonfinite = frozen.clone();
    nonfinite[1] = f32::NAN;
    assert!(exact_singleton(&[nonfinite], &frozen, &digest).is_err());

    let mut short = frozen.clone();
    short.pop();
    assert!(exact_singleton(&[short], &frozen, &digest).is_err());

    let mut changed_baseline = frozen.clone();
    changed_baseline[1] = -0.0;
    assert!(exact_singleton(&[changed_baseline.clone()], &changed_baseline, &digest,).is_err());
}

#[test]
fn exact_recovery_manifest_interval_preserves_bytes_and_rejects_rewrites() {
    let previous = b"{ \"kind\": \"startup\" }\n";
    let current = b"{ \"kind\": \"startup\" }\n{\"kind\":\"begin\"}\n{\"kind\":\"end\"}\n";
    assert_eq!(
        manifest_suffix(previous, current).unwrap(),
        b"{\"kind\":\"begin\"}\n{\"kind\":\"end\"}\n"
    );
    assert_eq!(manifest_suffix(previous, previous).unwrap(), b"");

    // Semantically equivalent JSON is not the same manifest byte prefix.
    assert!(manifest_suffix(previous, b"{\"kind\":\"startup\"}\n{\"kind\":\"begin\"}\n").is_err());
    assert!(manifest_suffix(previous, b"{}\n").is_err());
    assert!(manifest_suffix(previous, b"{ \"kind\": \"startup\" }\n{}").is_err());
    assert!(manifest_suffix(previous, b"{ \"kind\": \"startup\" }\n\n").is_err());
}

fn append_record(previous: &[u8], record: &Value) -> Vec<u8> {
    let mut bytes = previous.to_vec();
    serde_json::to_writer(&mut bytes, record).unwrap();
    bytes.push(b'\n');
    bytes
}

#[test]
fn exact_recovery_boundary_rejects_extra_records_wrong_digest_and_relabelling() {
    let previous = b"{ \"preceding\": true }\n";
    let digest = sha(b"recovery evidence");
    let record = json!({
        "acceptance_phase": policy::LIFECYCLE,
        "case_id": RECOVERY,
        "status": "executed",
        "pass": true,
        "details": {
            "evidence_path": format!("scheduler/{RECOVERY}/evidence.json"),
            "evidence_sha256": digest
        }
    });
    let current = append_record(previous, &record);
    assert!(one_record_extension(previous, &current, policy::LIFECYCLE, RECOVERY, &digest).is_ok());

    for (field, changed) in [
        ("acceptance_phase", json!(policy::SCHEDULER)),
        ("case_id", json!(DRAIN_NEXT)),
        ("status", json!("declared")),
        ("pass", json!(false)),
        ("owner_thread_joined", json!(true)),
    ] {
        let mut bad = record.clone();
        bad[field] = changed;
        assert!(
            one_record_extension(
                previous,
                &append_record(previous, &bad),
                policy::LIFECYCLE,
                RECOVERY,
                &digest
            )
            .is_err(),
            "{field}"
        );
    }

    let mut bad = record.clone();
    bad["details"]["evidence_sha256"] = json!(sha(b"new baseline"));
    assert!(one_record_extension(
        previous,
        &append_record(previous, &bad),
        policy::LIFECYCLE,
        RECOVERY,
        &digest
    )
    .is_err());

    let extra = append_record(&current, &record);
    assert!(one_record_extension(previous, &extra, policy::LIFECYCLE, RECOVERY, &digest).is_err());

    let rewritten = append_record(b"{\"preceding\":true}\n", &record);
    assert!(
        one_record_extension(previous, &rewritten, policy::LIFECYCLE, RECOVERY, &digest).is_err()
    );
}
