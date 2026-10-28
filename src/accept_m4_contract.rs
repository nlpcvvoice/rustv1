use anyhow::{ensure, Context, Result};
use rustv1::model::{BatchedExecution, EngineConfig, BATCHED_EXECUTION_POLICY};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

pub const WIDTH: usize = 2048;
pub const ROW_BYTES: usize = WIDTH * 4;
const JSON_LIMIT: usize = 32 * 1024 * 1024;
const MATRIX_LIMIT: usize = 64 * 1024 * 1024;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub root: String,
    pub path: String,
    pub sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    artifacts: BTreeMap<String, Artifact>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub id: String,
    pub text: String,
    pub ids: Vec<u32>,
    pub ids_sha256: String,
    pub golden_row: usize,
    pub golden_sha256: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shape {
    pub case_id: String,
    pub fixtures: Vec<usize>,
    #[serde(rename = "S")]
    pub s: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shapes {
    pub schema: String,
    pub batching: BatchedExecution,
    pub cases: Vec<Shape>,
}

pub struct Bundle {
    pub model_dir: PathBuf,
    pub input_sha256: String,
    pub artifacts: BTreeMap<String, Artifact>,
    pub paths: BTreeMap<String, PathBuf>,
    pub fixtures: Vec<Fixture>,
    pub singleton: Vec<Vec<f32>>,
    pub reference: Vec<Vec<f32>>,
    pub shapes: Shapes,
    pub candidates: BTreeMap<(Vec<usize>, usize), Vec<f32>>,
    pub audit_identity: String,
    pub engine: EngineConfig,
}

pub fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn digest_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn label_valid(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 80
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
}

pub fn file_sha(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub fn bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    ensure!(file.metadata()?.len() <= limit as u64, "oversized file");
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= limit, "file grew beyond bound");
    Ok(bytes)
}

pub fn value(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&bounded(path, JSON_LIMIT)?)?)
}

pub fn unresolved(value: &Value) -> bool {
    match value {
        Value::String(text) => [
            "RUNTIME",
            "FILLED_BY_INTEGRATOR",
            "<SHA256:",
            "PENDING_ROUNDTRIP",
        ]
        .iter()
        .any(|needle| text.contains(needle)),
        Value::Array(values) => values.iter().any(unresolved),
        Value::Object(values) => values
            .iter()
            .any(|(key, value)| key.contains("RUNTIME") || unresolved(value)),
        _ => false,
    }
}

pub fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

pub fn row_bytes(row: &[f32]) -> Vec<u8> {
    row.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn ids_sha(ids: &[u32]) -> String {
    sha(&ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}

pub fn norm(row: &[f32]) -> Result<f64> {
    ensure!(row.len() == WIDTH, "embedding width");
    ensure!(row.iter().all(|x| x.is_finite()), "nonfinite embedding");
    let n = row
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(
        n.is_finite() && (n - 1.0).abs() <= 1e-5,
        "FP64 norm gate: {n}"
    );
    Ok(n)
}

pub fn compare(actual: &[f32], expected: &[f32]) -> Result<Value> {
    let an = norm(actual)?;
    let en = norm(expected)?;
    let dot = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum::<f64>();
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .fold(0.0f64, f64::max);
    let cosine = dot / (an * en);
    ensure!(
        cosine.is_finite() && cosine > 0.999 && max_abs <= 0.01,
        "numerical gate: cosine={cosine}, max_abs={max_abs}"
    );
    Ok(json!({"cosine": cosine, "max_abs": max_abs}))
}

pub fn exact(actual: &[Vec<f32>], expected: &[Vec<f32>]) -> Result<()> {
    ensure!(actual.len() == expected.len(), "exact row cardinality");
    for (actual, expected) in actual.iter().zip(expected) {
        norm(actual)?;
        norm(expected)?;
        ensure!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "exact F32 repeat mismatch"
        );
    }
    Ok(())
}

fn rows(path: &Path) -> Result<Vec<Vec<f32>>> {
    let bytes = bounded(path, MATRIX_LIMIT)?;
    ensure!(
        !bytes.is_empty() && bytes.len() % ROW_BYTES == 0,
        "F32 matrix size"
    );
    bytes
        .as_chunks::<ROW_BYTES>()
        .0
        .iter()
        .map(|bytes| {
            let row: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            norm(&row)?;
            Ok(row)
        })
        .collect()
}

fn path_under(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    ensure!(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "artifact path must be relative without traversal"
    );
    let root = root.canonicalize()?;
    let path = root.join(relative).canonicalize()?;
    ensure!(
        path.starts_with(&root) && path.is_file(),
        "artifact escapes root"
    );
    Ok(path)
}

impl Bundle {
    pub fn path(&self, role: &str) -> Result<&Path> {
        self.paths
            .get(role)
            .map(PathBuf::as_path)
            .with_context(|| format!("missing artifact {role}"))
    }

    pub fn candidate(&self, indices: &[usize], position: usize) -> Result<&[f32]> {
        self.candidates
            .get(&(indices.to_vec(), position))
            .map(Vec::as_slice)
            .context("missing shape-specific candidate row")
    }

    pub fn canonical(&self, fixture: usize, position: usize) -> Result<&[f32]> {
        ensure!(position < 2, "canonical position");
        let indices = if position == 0 {
            vec![fixture, 6]
        } else {
            vec![6, fixture]
        };
        self.candidate(&indices, position)
    }

    pub fn recheck_bindings(&self) -> Result<()> {
        for (role, artifact) in &self.artifacts {
            ensure!(
                file_sha(self.path(role)?)? == artifact.sha256,
                "artifact changed during pre-pass: {role}"
            );
        }
        Ok(())
    }
}

pub fn load(model_dir: &Path, input_path: &Path, expected_sha: &str) -> Result<Bundle> {
    ensure!(digest_valid(expected_sha), "pre-pass input digest format");
    let raw = bounded(input_path, JSON_LIMIT)?;
    ensure!(sha(&raw) == expected_sha, "pre-pass input digest mismatch");
    let input_value: Value = serde_json::from_slice(&raw)?;
    ensure!(!unresolved(&input_value), "unresolved pre-pass input");
    let input: Input = serde_json::from_value(input_value)?;
    ensure!(
        input.schema == "rustv1.accept_m4_prepass.v1",
        "pre-pass input schema"
    );

    let model_dir = model_dir.canonicalize()?;
    let repo = std::env::current_dir()?.canonicalize()?;
    let parent = input_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut paths = BTreeMap::new();
    for (role, artifact) in &input.artifacts {
        ensure!(digest_valid(&artifact.sha256), "artifact digest: {role}");
        let root = match artifact.root.as_str() {
            "model" => model_dir.as_path(),
            "repo" => repo.as_path(),
            "policy" => parent,
            _ => anyhow::bail!("unknown artifact root"),
        };
        let path = path_under(root, &artifact.path)?;
        ensure!(file_sha(&path)? == artifact.sha256, "artifact hash: {role}");
        paths.insert(role.clone(), path);
    }
    let required = |role: &str| -> Result<&Path> {
        paths
            .get(role)
            .map(PathBuf::as_path)
            .with_context(|| format!("missing artifact {role}"))
    };
    for role in [
        "config",
        "engine",
        "tokenizer",
        "cargo_lock",
        "feature_tree",
        "audit",
        "reference_provenance",
        "fixtures_exec",
        "fixture_goldens",
        "fixture_reference",
        "batch_shapes",
        "batch_goldens",
        "batch_golden_index",
        "batch_export_provenance",
        "batch_export_manifest",
    ] {
        required(role)?;
    }
    for role in ["config", "engine", "tokenizer"] {
        ensure!(
            required(role)? == model_dir.join(format!("{role}.json")).canonicalize()?,
            "model artifact path mismatch: {role}"
        );
    }
    ensure!(
        required("cargo_lock")? == repo.join("Cargo.lock").canonicalize()?,
        "Cargo.lock path mismatch"
    );
    let mut found_weight = false;
    for entry in fs::read_dir(&model_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json") {
            let path = entry.path().canonicalize()?;
            ensure!(
                input.artifacts.iter().any(|(role, artifact)| {
                    role.starts_with("weight:")
                        && artifact.root == "model"
                        && paths.get(role) == Some(&path)
                }),
                "unbound checkpoint file: {name}"
            );
            found_weight |= name.ends_with(".safetensors");
        }
    }
    ensure!(found_weight, "missing checkpoint weights");

    let engine: EngineConfig = serde_json::from_slice(&bounded(required("engine")?, JSON_LIMIT)?)?;
    let shapes_value = value(required("batch_shapes")?)?;
    ensure!(!unresolved(&shapes_value), "unresolved shape plan");
    let shapes: Shapes = serde_json::from_value(shapes_value)?;
    ensure!(shapes.schema == "rustv1.batch_shapes.v1", "shape schema");
    shapes.batching.validate_engine(&engine)?;
    ensure!(
        engine.max_len == 1024 && shapes.batching.max_batch_size == 32,
        "full numerical qualification requires max_len=1024 and B ceiling 32"
    );
    ensure!(shapes.batching.fits(32, 1024), "B32/S1024 must fit");
    ensure!(shapes.cases.len() <= 128, "too many declared shapes");

    let fixtures: Vec<Fixture> =
        serde_json::from_slice(&bounded(required("fixtures_exec")?, JSON_LIMIT)?)?;
    ensure!(fixtures.len() == 22, "fixture count");
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(&bounded(required("tokenizer")?, JSON_LIMIT)?)
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
    ensure!(
        tokenizer.get_padding().is_none() && tokenizer.get_truncation().is_none(),
        "tokenizer padding/truncation must be disabled"
    );

    let golden_matrix = rows(required("fixture_goldens")?)?;
    let reference = rows(required("fixture_reference")?)?;
    ensure!(
        reference.len() == 22,
        "reference must be in 22-fixture order"
    );
    let mut names = BTreeSet::new();
    let mut singleton = Vec::new();
    for (i, fixture) in fixtures.iter().enumerate() {
        ensure!(names.insert(&fixture.id), "duplicate fixture ID");
        ensure!(!fixture.text.trim().is_empty(), "empty fixture text");
        ensure!(
            !fixture.ids.is_empty()
                && fixture.ids.len() <= 1024
                && fixture.ids.iter().all(|&id| id < 131072 && id != 11),
            "invalid fixture token IDs"
        );
        ensure!(
            ids_sha(&fixture.ids) == fixture.ids_sha256,
            "fixture token digest"
        );
        let encoded = tokenizer
            .encode(fixture.text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
        ensure!(
            encoded.get_ids() == fixture.ids.as_slice(),
            "fixture text/IDs"
        );
        let golden = golden_matrix
            .get(fixture.golden_row)
            .context("golden row")?;
        ensure!(
            sha(&row_bytes(golden)) == fixture.golden_sha256,
            "frozen singleton digest"
        );
        compare(golden, &reference[i])?;
        singleton.push(golden.clone());
    }
    ensure!(
        fixtures[..7]
            .iter()
            .map(|f| f.ids.len())
            .collect::<Vec<_>>()
            == vec![1, 2, 255, 256, 257, 1023, 1024],
        "boundary fixture lengths"
    );

    let mut case_names = BTreeSet::new();
    let mut layouts = BTreeSet::new();
    for shape in &shapes.cases {
        ensure!(
            label_valid(&shape.case_id)
                && !shape.case_id.starts_with("canonical-")
                && case_names.insert(&shape.case_id),
            "invalid/duplicate/reserved shape case ID"
        );
        ensure!(
            !shape.fixtures.is_empty() && shape.fixtures.iter().all(|&i| i < 22),
            "invalid shape fixtures"
        );
        let maximum = shape.fixtures.iter().map(|&i| fixtures[i].ids.len()).max();
        ensure!(
            maximum == Some(shape.s),
            "serving shape S must equal max length"
        );
        ensure!(
            shapes.batching.fits(shape.fixtures.len(), shape.s),
            "shape exceeds capacity"
        );
        layouts.insert(shape.fixtures.clone());
    }
    for b in [2, 4, 8, 16, 32] {
        for fixture in [0, 6] {
            ensure!(
                layouts.contains(&vec![fixture; b]),
                "missing homogeneous fixture {fixture} B={b}"
            );
        }
    }
    for layout in [
        vec![0, 1, 2, 3, 4, 5, 6],
        vec![6, 5, 4, 3, 2, 1, 0],
        vec![0, 6, 0, 6],
    ] {
        ensure!(
            layouts.contains(&layout),
            "missing required mixed/permutation layout"
        );
    }

    let audit = value(required("audit")?)?;
    let audit_artifact = &input.artifacts["audit"];
    ensure!(audit_artifact.root == "repo", "audit must be repo-relative");
    ensure!(
        !unresolved(&audit)
            && audit["schema"] == "rustv1.cuda_math_source_audit.v1"
            && audit["status"] == "accepted"
            && audit["signed"] == true
            && audit["verification"]["mechanical_region_boundary_verification_completed"] == true
            && audit["contract_id"] == "b200-independent-F-v2"
            && audit["intended_path"] == audit_artifact.path
            && audit["scope"]["execution_policy"] == BATCHED_EXECUTION_POLICY
            && audit["scope"]["precision_profile"] == "F"
            && audit["scope"]["maximum_physical_b"]
                .as_u64()
                .is_some_and(|b| b >= 32)
            && audit["scope"]["sequence_length_max"]
                .as_u64()
                .is_some_and(|s| s >= 1024)
            && audit["scope"]["physical_sequence_batch_size"] != 1
            && audit["cargo_lock_sha256"] == input.artifacts["cargo_lock"].sha256
            && audit["cargo_tree_features"]["sha256"] == input.artifacts["feature_tree"].sha256,
        "active audit scope/bindings"
    );
    let provenance = value(required("reference_provenance")?)?;
    ensure!(
        provenance.is_object() && !unresolved(&provenance),
        "invalid reference provenance artifact"
    );
    // The externally designated input SHA is the trust anchor. This does not
    // interpret an unspecified reference-provenance signature format.
    let audit_identity = format!("sha256:{}:{}", audit_artifact.sha256, audit_artifact.path);

    let mut bundle = Bundle {
        model_dir,
        input_sha256: sha(&raw),
        artifacts: input.artifacts,
        paths,
        fixtures,
        singleton,
        reference,
        shapes,
        candidates: BTreeMap::new(),
        audit_identity,
        engine,
    };
    load_candidates(&mut bundle)?;
    Ok(bundle)
}

fn load_candidates(bundle: &mut Bundle) -> Result<()> {
    let index = value(bundle.path("batch_golden_index")?)?;
    let matrix = rows(bundle.path("batch_goldens")?)?;
    ensure!(
        index["schema"] == "rustv1.batch_candidate_goldens.v1"
            && index["execution_policy"] == BATCHED_EXECUTION_POLICY
            && index["profile"] == "F"
            && index["width"] == WIDTH
            && index["dtype"] == "F32"
            && index["byte_order"] == "little"
            && index["canonical_row_count"] == 44
            && index["row_count"] == matrix.len()
            && index["matrix_bytes"] == matrix.len() * ROW_BYTES
            && index["matrix_sha256"] == bundle.artifacts["batch_goldens"].sha256
            && index["fixtures_sha256"] == bundle.artifacts["fixtures_exec"].sha256
            && index["shape_plan_sha256"] == bundle.artifacts["batch_shapes"].sha256
            && index["export_provenance_sha256"]
                == bundle.artifacts["batch_export_provenance"].sha256
            && index["executor_manifest_sha256"]
                == bundle.artifacts["batch_export_manifest"].sha256
            && index["declared_case_count"] == bundle.shapes.cases.len(),
        "batch candidate index binding"
    );
    let provenance = value(bundle.path("batch_export_provenance")?)?;
    ensure!(
        provenance["schema"] == "rustv1.batch_golden_export_provenance.v1"
            && provenance["execution_policy"] == BATCHED_EXECUTION_POLICY
            && provenance["profile"] == "F"
            && provenance["audit_identity"] == bundle.audit_identity
            && provenance["cargo_lock_sha256"] == bundle.artifacts["cargo_lock"].sha256
            && provenance["fixtures_sha256"] == bundle.artifacts["fixtures_exec"].sha256
            && provenance["shape_plan_sha256"] == bundle.artifacts["batch_shapes"].sha256
            && provenance["batching"] == serde_json::to_value(&bundle.shapes.batching)?,
        "batch export provenance"
    );
    for (role, artifact) in &bundle.artifacts {
        if matches!(role.as_str(), "config" | "engine" | "tokenizer") || role.starts_with("weight:")
        {
            let name = bundle
                .path(role)?
                .file_name()
                .context("artifact filename")?;
            let name = name.to_str().context("non-UTF8 artifact filename")?;
            ensure!(
                provenance["model_files"][name] == artifact.sha256,
                "export model binding: {name}"
            );
        }
    }

    let mut expected: Vec<(String, Vec<usize>, usize)> = Vec::new();
    for fixture in 0..22 {
        expected.push((format!("canonical-47-{fixture}-0"), vec![fixture, 6], 0));
        expected.push((format!("canonical-47-{fixture}-1"), vec![6, fixture], 1));
    }
    for shape in &bundle.shapes.cases {
        for position in 0..shape.fixtures.len() {
            expected.push((shape.case_id.clone(), shape.fixtures.clone(), position));
        }
    }
    let entries = index["rows"].as_array().context("candidate rows")?;
    ensure!(
        entries.len() == expected.len() && matrix.len() == expected.len(),
        "candidate row coverage"
    );
    for (matrix_row, ((case, indices, position), entry)) in expected.iter().zip(entries).enumerate()
    {
        let lengths: Vec<usize> = indices
            .iter()
            .map(|&i| bundle.fixtures[i].ids.len())
            .collect();
        let digests: Vec<&str> = indices
            .iter()
            .map(|&i| bundle.fixtures[i].ids_sha256.as_str())
            .collect();
        let fixture = &bundle.fixtures[indices[*position]];
        ensure!(
            entry["matrix_row"] == matrix_row
                && entry["byte_offset"] == matrix_row * ROW_BYTES
                && entry["byte_length"] == ROW_BYTES
                && entry["case_id"] == *case
                && entry["ordered_fixture_indices"] == json!(indices)
                && entry["ordered_token_digests"] == json!(digests)
                && entry["valid_lengths"] == json!(lengths)
                && entry["B"] == indices.len()
                && entry["S"] == *lengths.iter().max().context("empty candidate")?
                && entry["physical_row"] == *position
                && entry["fixture_index"] == indices[*position]
                && entry["fixture_id"] == fixture.id
                && entry["execution_policy"] == BATCHED_EXECUTION_POLICY
                && entry["route"] == "IndependentExecutor::forward_batch"
                && entry["output_sha256"] == sha(&row_bytes(&matrix[matrix_row])),
            "candidate key mismatch at row {matrix_row}"
        );
        let key = (indices.clone(), *position);
        if let Some(previous) = bundle.candidates.insert(key, matrix[matrix_row].clone()) {
            exact(&[previous], &[matrix[matrix_row].clone()])?;
        }
    }
    Ok(())
}
