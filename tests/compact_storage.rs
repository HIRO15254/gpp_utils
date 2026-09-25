//! Stored run results and incomplete markers are compact single-line JSON with
//! a packed partition pool; experiment and graph files stay pretty-printed.
use gpp_utils::{
    ExperimentSpec, compile_experiment,
    experiment::{plan::compile_experiment_with_versions, result::RunTermination},
    fitness::{FitnessFactory, FitnessRegistry, VertexFitness},
    graph_partition::{Graph, PartitionState},
    optimization::CancellationToken,
    storage::{RuntimeOptions, incomplete_path, inspect, read_result, result_path, run_batch},
};
use serde::Serialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, sync::Arc};

/// Independent decoder of the documented packing: vertex `v` is bit `v % 8`
/// of byte `v / 8`, bytes are two lowercase hexadecimal digits, and the
/// unused high bits of the last byte are zero.
fn unpack(hex: &str, length: usize) -> Vec<bool> {
    assert_eq!(hex.len(), 2 * length.div_ceil(8), "digit count of {hex}");
    assert!(
        hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
        "lowercase hex {hex}"
    );
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();
    for v in length..8 * bytes.len() {
        assert_eq!((bytes[v / 8] >> (v % 8)) & 1, 0, "padding bit {v} of {hex}");
    }
    (0..length)
        .map(|v| (bytes[v / 8] >> (v % 8)) & 1 == 1)
        .collect()
}

/// Asserts one JSON line followed by `\n`, without whitespace outside strings.
fn assert_compact(bytes: &[u8]) {
    assert_eq!(bytes.last(), Some(&b'\n'), "missing trailing newline");
    let body = &bytes[..bytes.len() - 1];
    let (mut in_string, mut escaped) = (false, false);
    for &byte in body {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
        } else if byte == b'"' {
            in_string = true;
        } else {
            assert!(
                !byte.is_ascii_whitespace(),
                "whitespace outside a JSON string"
            );
        }
    }
    assert!(!in_string && !body.contains(&b'\n'));
    serde_json::from_slice::<Value>(body).unwrap();
}

fn assert_pretty(bytes: &[u8]) {
    assert_eq!(bytes.last(), Some(&b'\n'));
    assert!(bytes.iter().filter(|&&b| b == b'\n').count() > 1);
    assert!(bytes.windows(3).any(|w| w == b"\n  "));
}

fn options(root: &Path) -> RuntimeOptions {
    RuntimeOptions {
        root: root.into(),
        threads: 2,
        ..Default::default()
    }
}

fn spec(neighborhoods: &[&str], node_counts: &[usize], solvers: Value) -> ExperimentSpec {
    serde_json::from_value(json!({
        "schema_version": 1, "run_seeds": [1, 2], "neighborhoods": neighborhoods,
        "budget": {"max_steps": 200},
        "measurement": {
            "schedule": "logarithmic", "basin": "real", "best_basin": true,
            "max_basin_steps": 50, "diagnostics": true
        },
        "graphs": [{
            "kind": "random", "node_counts": node_counts,
            "expected_degrees": [3.0], "seeds": [5]
        }],
        "solvers": solvers
    }))
    .unwrap()
}

fn sa() -> Value {
    json!([{"kind": "sa", "temperatures": [1.0], "smoothing": [{"kind": "none"}]}])
}

/// Field order of the stored graph file written before and after the
/// compact result format; graph files must remain byte-identical.
#[derive(Serialize)]
struct GraphFile<'a> {
    schema_version: u32,
    node_count: usize,
    edges: &'a [[usize; 2]],
    content_hash: String,
}

#[test]
fn results_are_compact_with_packed_partitions_and_inputs_stay_pretty() {
    let plan = compile_experiment(spec(&["flip", "swap"], &[14, 16], sa())).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let summary = run_batch(
        &plan,
        &options(temp.path()),
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    assert_eq!(summary.completed, plan.jobs.len());
    let graphs: BTreeMap<&str, Graph> = plan
        .jobs
        .iter()
        .map(|job| {
            let graph = Graph::generate(&job.condition.graph, &CancellationToken::new());
            (job.graph_id.as_str(), graph.unwrap())
        })
        .collect();
    assert_eq!(graphs.len(), 2);
    for job in &plan.jobs {
        let graph = &graphs[job.graph_id.as_str()];
        let path = result_path(temp.path(), job);
        let bytes = std::fs::read(&path).unwrap();
        assert_compact(&bytes);
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        let pool = value["partitions"].as_object().unwrap();
        assert_eq!(
            pool.keys().map(String::as_str).collect::<Vec<_>>(),
            ["hex", "length"]
        );
        assert_eq!(pool["length"], json!(graph.node_count()));
        let result = read_result(&path, graph, &job.condition).unwrap();
        let hex = pool["hex"].as_array().unwrap();
        assert_eq!(hex.len(), result.partitions.len());
        for (text, partition) in hex.iter().zip(&result.partitions) {
            assert_eq!(
                unpack(text.as_str().unwrap(), graph.node_count()),
                *partition
            );
        }
        // Only the partition representation changed: the remaining fields
        // keep their existing names and order.
        let body = std::str::from_utf8(&bytes).unwrap();
        let keys = [
            "\"schema_version\":1,",
            "\"attempt_id\":",
            "\"termination\":\"step_limit\",",
            "\"completed_steps\":200,",
            "\"elapsed_ms\":",
            ",\"partitions\":{\"length\":",
            "]},\"final_solution\":",
            ",\"best_solution\":",
            ",\"best_step\":",
            ",\"records\":[{\"step\":0,\"current_solution\":0,\"best_solution\":0,",
            "}],\"diagnostics\":{\"applied_moves\":",
        ];
        let mut offset = 0;
        for key in keys {
            let found = body[offset..]
                .find(key)
                .unwrap_or_else(|| panic!("{key} after byte {offset}"));
            offset += found + key.len();
        }
    }
    for (graph_id, graph) in &graphs {
        let path = temp.path().join("graphs").join(format!("{graph_id}.json"));
        let bytes = std::fs::read(path).unwrap();
        assert_pretty(&bytes);
        let mut expected = serde_json::to_vec_pretty(&GraphFile {
            schema_version: 1,
            node_count: graph.node_count(),
            edges: graph.edges(),
            content_hash: graph.content_hash(),
        })
        .unwrap();
        expected.push(b'\n');
        assert_eq!(bytes, expected);
    }
    let experiment = std::fs::read(
        temp.path()
            .join("batches")
            .join(&plan.batch_id)
            .join("experiment.json"),
    )
    .unwrap();
    assert_pretty(&experiment);
    let mut expected = serde_json::to_vec_pretty(&plan.experiment).unwrap();
    expected.push(b'\n');
    assert_eq!(experiment, expected);
}

struct CancellingFitness(CancellationToken);
impl VertexFitness for CancellingFitness {
    fn values(&self, graph: &Graph, _: &PartitionState) -> gpp_utils::error::Result<Vec<f64>> {
        self.0.cancel();
        Ok(vec![0.0; graph.node_count()])
    }
}
struct CancellingFactory(CancellationToken);
impl FitnessFactory for CancellingFactory {
    fn version(&self) -> &str {
        "cancel-v1"
    }
    fn validate(&self, _: &Value) -> gpp_utils::error::Result<()> {
        Ok(())
    }
    fn create(&self, _: &Value) -> gpp_utils::error::Result<Box<dyn VertexFitness>> {
        Ok(Box::new(CancellingFitness(self.0.clone())))
    }
}

#[test]
fn cancelled_marker_is_compact_and_its_partial_result_is_restored() {
    let mut spec = spec(
        &["flip"],
        &[13],
        json!([{"kind": "eo", "taus": [1.0], "fitnesses": [{"kind": "cancel", "params": {}}]}]),
    );
    spec.run_seeds = vec![3];
    let plan = compile_experiment_with_versions(
        spec,
        &BTreeMap::from([("cancel".into(), "cancel-v1".into())]),
    )
    .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let mut registry = FitnessRegistry::default();
    registry.register("cancel", Arc::new(CancellingFactory(cancel.clone())));
    let single = RuntimeOptions {
        threads: 1,
        ..options(temp.path())
    };
    let summary = run_batch(&plan, &single, &cancel, &registry, &|_| {}).unwrap();
    assert_eq!(summary.cancelled, 1);
    let job = &plan.jobs[0];
    assert!(!result_path(temp.path(), job).exists());
    let bytes = std::fs::read(incomplete_path(temp.path(), job)).unwrap();
    assert_compact(&bytes);
    let marker: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(marker["status"], "cancelled");
    let stored = &marker["partial_result"];
    assert_eq!(stored["partitions"]["length"], 13);
    let hex = stored["partitions"]["hex"].as_array().unwrap();
    assert!(!hex.is_empty());

    let rows = inspect(&plan, temp.path(), true).unwrap();
    assert_eq!(rows[0].status, "cancelled");
    let restored = rows[0].result.as_ref().expect("restored partial result");
    assert_eq!(restored.termination, RunTermination::Cancelled);
    assert_eq!(restored.attempt_id, marker["attempt_id"].as_str().unwrap());
    let decoded: Vec<Vec<bool>> = hex
        .iter()
        .map(|text| unpack(text.as_str().unwrap(), 13))
        .collect();
    assert_eq!(restored.partitions, decoded);
    let mut value = serde_json::to_value(restored).unwrap();
    for key in ["schema_version", "attempt_id", "termination"] {
        value.as_object_mut().unwrap().remove(key);
    }
    assert_eq!(&value, stored);
    assert!(
        inspect(&plan, temp.path(), false).unwrap()[0]
            .result
            .is_none()
    );
}

#[test]
fn malformed_packed_partitions_are_rejected_when_reading_results() {
    let mut spec = spec(&["flip"], &[13], sa());
    spec.run_seeds = vec![4];
    let plan = compile_experiment(spec).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let normal = options(temp.path());
    let run = || {
        run_batch(
            &plan,
            &normal,
            &CancellationToken::new(),
            &FitnessRegistry::default(),
            &|_| {},
        )
        .unwrap()
    };
    assert_eq!(run().completed, 1);
    let job = &plan.jobs[0];
    let graph = Graph::generate(&job.condition.graph, &CancellationToken::new()).unwrap();
    let path = result_path(temp.path(), job);
    let original = std::fs::read(&path).unwrap();
    let value: Value = serde_json::from_slice(&original).unwrap();
    let count = value["partitions"]["hex"].as_array().unwrap().len();
    type Edit = fn(&mut Value);
    let cases: [(Edit, &str); 10] = [
        // Valid 14- and 16-vertex encodings of the 13-vertex pool: only the
        // graph-aware validation can reject them.
        (|p| p["length"] = json!(14), "invalid partition length"),
        (|p| p["length"] = json!(16), "invalid partition length"),
        (|p| p["hex"][0] = json!("ff1"), "found 3"),
        (|p| p["hex"][0] = json!("ff1f00"), "found 6"),
        (|p| p["hex"][0] = json!("FF1F"), "uppercase"),
        (|p| p["hex"][0] = json!("fg1f"), "non-hexadecimal"),
        (|p| p["hex"][0] = json!("ff3f"), "padding"),
        (|p| p["hex"][0] = json!(8191), "invalid type"),
        (|p| p["bits"] = json!(13), "unknown field `bits`"),
        (
            |p| *p = json!([vec![true; 13], vec![false; 13]]),
            "invalid type",
        ),
    ];
    for (edit, message) in cases {
        let mut bad = value.clone();
        edit(&mut bad["partitions"]);
        std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
        let error = format!(
            "{:#}",
            read_result(&path, &graph, &job.condition).unwrap_err()
        );
        assert!(error.contains(message), "{message}: {error}");
    }
    let mut missing = value.clone();
    missing["partitions"].as_object_mut().unwrap().remove("hex");
    std::fs::write(&path, serde_json::to_vec(&missing).unwrap()).unwrap();
    let error = format!(
        "{:#}",
        read_result(&path, &graph, &job.condition).unwrap_err()
    );
    assert!(error.contains("missing field `hex`"), "{error}");
    // The pool size is unchanged by every edit above.
    assert!(count > 1);

    // A malformed pool is ordinary corruption: preserved without recovery,
    // then backed up and recomputed on request.
    let corrupt = std::fs::read(&path).unwrap();
    assert_eq!(run().failed, 1);
    assert_eq!(std::fs::read(&path).unwrap(), corrupt);
    let recover = RuntimeOptions {
        recover_corrupt: true,
        ..normal.clone()
    };
    let summary = run_batch(
        &plan,
        &recover,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    assert_eq!(summary.completed, 1);
    let recomputed = read_result(&path, &graph, &job.condition).unwrap();
    let before: gpp_utils::RunResult = serde_json::from_slice(&original).unwrap();
    assert_eq!(recomputed.partitions, before.partitions);
}
