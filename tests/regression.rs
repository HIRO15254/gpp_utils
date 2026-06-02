//! 回帰テスト: 採取済み baseline と現在の実装の出力を照合し、意図しない
//! ドリフトを検出する。
//!
//! `tests/data/regression_baseline.json` は現行実装で
//! `cargo run --release --example gen_baseline > tests/data/regression_baseline.json`
//! を走らせて生成する。アルゴリズムを意図的に変更したときは baseline も
//! 再生成すること。
//!
//! 検証:
//! - **`final_partition` がビット完全一致** (Vec<bool> の == 比較)
//! - `records` 内の f64 値は 1e-12 許容誤差で比較

use gpp_utils::graph_spec::{GraphSpec, StoredGraph};
use gpp_utils::run_config::{RunConfig, SmoothingSpec, SolverSpec};
use gpp_utils::run_executor::execute;
use serde::Deserialize;

#[derive(Deserialize)]
struct BaselineEntry {
    smoothing: String,
    seed: u64,
    final_partition: Vec<bool>,
    records: Vec<(usize, f64, f64, f64, f64, f64, f64)>,
}

/// τ-EO の baseline エントリ。
#[derive(Deserialize)]
struct EoEntry {
    tau: f64,
    seed: u64,
    final_partition: Vec<bool>,
    records: Vec<(usize, f64, f64, f64, f64, f64, f64)>,
}

#[derive(Deserialize)]
struct Baseline {
    graph_spec: GraphSpec,
    log10_iterations: u32,
    theta: Option<f64>,
    entries: Vec<BaselineEntry>,
    /// EO（swap版）エントリ（加法的拡張。古い baseline には無いので default = 空）。
    #[serde(default)]
    eo_entries: Vec<EoEntry>,
    /// EO（flip版）エントリ（加法的拡張）。
    #[serde(default)]
    eoflip_entries: Vec<EoEntry>,
}

fn load_baseline() -> Baseline {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("regression_baseline.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("baseline file not found at {}", path.display()));
    serde_json::from_str(&text).expect("failed to parse baseline JSON")
}

fn parse_smoothing(label: &str) -> SmoothingSpec {
    match label {
        "none" => SmoothingSpec::None,
        "kavg8" => SmoothingSpec::KAverage(8),
        "rkavg8" => SmoothingSpec::RandomKAverage(8),
        "wavg0.5" => SmoothingSpec::WeightedAverage(0.5),
        other => panic!("unknown smoothing label: {}", other),
    }
}

#[test]
fn regression_final_partition_bitwise_match() {
    let baseline = load_baseline();
    let stored = StoredGraph::generate(baseline.graph_spec);
    let prob = stored.problem();

    let mut total = 0;
    let mut mismatched_records = Vec::new();

    for entry in &baseline.entries {
        let mut cfg = RunConfig::new("regression");
        cfg.theta = baseline.theta;
        cfg.log10_iterations = baseline.log10_iterations;
        cfg.smoothing = parse_smoothing(&entry.smoothing);

        let result = execute(baseline.graph_spec, &cfg, &prob, entry.seed);

        // final_partition のビット完全一致を検証
        assert_eq!(
            result.final_partition, entry.final_partition,
            "final_partition mismatch: smoothing={}, seed={}",
            entry.smoothing, entry.seed
        );

        // records は 1e-12 許容誤差で参考比較
        assert_eq!(
            result.records.len(),
            entry.records.len(),
            "records length mismatch: smoothing={}, seed={}",
            entry.smoothing, entry.seed
        );
        for (i, (new, old)) in result.records.iter().zip(entry.records.iter()).enumerate() {
            let fields: [(f64, f64, &str); 6] = [
                (new.current_smoothed, old.1, "current_smoothed"),
                (new.current_real, old.2, "current_real"),
                (new.basin_smoothed_from_smoothed, old.3, "basin_smoothed_from_smoothed"),
                (new.basin_real_from_smoothed, old.4, "basin_real_from_smoothed"),
                (new.basin_smoothed_from_real, old.5, "basin_smoothed_from_real"),
                (new.basin_real_from_real, old.6, "basin_real_from_real"),
            ];
            assert_eq!(new.step, old.0, "step mismatch at record {}", i);
            for &(n, o, name) in &fields {
                let diff = (n - o).abs();
                let tol = 1e-12 * (1.0 + n.abs().max(o.abs()));
                if diff > tol {
                    mismatched_records.push(format!(
                        "smoothing={}, seed={}, rec={}, field={}, new={}, old={}, diff={}",
                        entry.smoothing, entry.seed, i, name, n, o, diff
                    ));
                }
            }
        }
        total += 1;
    }

    assert!(
        mismatched_records.is_empty(),
        "f64 record mismatches ({} entries):\n{}",
        mismatched_records.len(),
        mismatched_records.join("\n")
    );

    assert_eq!(total, 80, "expected 4 smoothings × 20 seeds = 80 entries");
}

#[test]
fn regression_eo_final_partition_bitwise_match() {
    let baseline = load_baseline();
    // 古い baseline（eo_entries 無し）では何も検証しない（後方互換）。
    if baseline.eo_entries.is_empty() {
        return;
    }
    let stored = StoredGraph::generate(baseline.graph_spec);
    let prob = stored.problem();

    let mut total = 0;
    let mut mismatched_records = Vec::new();

    for entry in &baseline.eo_entries {
        let mut cfg = RunConfig::new("regression-eo");
        cfg.theta = None;
        cfg.smoothing = SmoothingSpec::None;
        cfg.log10_iterations = baseline.log10_iterations;
        cfg.solver = SolverSpec::Eo { tau: entry.tau };

        let result = execute(baseline.graph_spec, &cfg, &prob, entry.seed);

        assert_eq!(
            result.final_partition, entry.final_partition,
            "EO final_partition mismatch: tau={}, seed={}",
            entry.tau, entry.seed
        );
        assert_eq!(
            result.records.len(),
            entry.records.len(),
            "EO records length mismatch: tau={}, seed={}",
            entry.tau, entry.seed
        );
        for (i, (new, old)) in result.records.iter().zip(entry.records.iter()).enumerate() {
            let fields: [(f64, f64, &str); 6] = [
                (new.current_smoothed, old.1, "current_smoothed"),
                (new.current_real, old.2, "current_real"),
                (new.basin_smoothed_from_smoothed, old.3, "basin_smoothed_from_smoothed"),
                (new.basin_real_from_smoothed, old.4, "basin_real_from_smoothed"),
                (new.basin_smoothed_from_real, old.5, "basin_smoothed_from_real"),
                (new.basin_real_from_real, old.6, "basin_real_from_real"),
            ];
            assert_eq!(new.step, old.0, "EO step mismatch at record {}", i);
            for &(n, o, name) in &fields {
                let diff = (n - o).abs();
                let tol = 1e-12 * (1.0 + n.abs().max(o.abs()));
                if diff > tol {
                    mismatched_records.push(format!(
                        "tau={}, seed={}, rec={}, field={}, new={}, old={}, diff={}",
                        entry.tau, entry.seed, i, name, n, o, diff
                    ));
                }
            }
        }
        total += 1;
    }

    assert!(
        mismatched_records.is_empty(),
        "EO f64 record mismatches ({} entries):\n{}",
        mismatched_records.len(),
        mismatched_records.join("\n")
    );

    assert_eq!(total, 40, "expected 2 taus × 20 seeds = 40 EO entries");
}

#[test]
fn regression_eoflip_final_partition_bitwise_match() {
    let baseline = load_baseline();
    if baseline.eoflip_entries.is_empty() {
        return;
    }
    let stored = StoredGraph::generate(baseline.graph_spec);
    let prob = stored.problem();

    let mut total = 0;
    let mut mismatched_records = Vec::new();

    for entry in &baseline.eoflip_entries {
        let mut cfg = RunConfig::new("regression-eoflip");
        cfg.theta = None;
        cfg.smoothing = SmoothingSpec::None;
        cfg.log10_iterations = baseline.log10_iterations;
        cfg.solver = SolverSpec::EoFlip { tau: entry.tau };

        let result = execute(baseline.graph_spec, &cfg, &prob, entry.seed);

        assert_eq!(
            result.final_partition, entry.final_partition,
            "EoFlip final_partition mismatch: tau={}, seed={}",
            entry.tau, entry.seed
        );
        assert_eq!(
            result.records.len(),
            entry.records.len(),
            "EoFlip records length mismatch: tau={}, seed={}",
            entry.tau, entry.seed
        );
        for (i, (new, old)) in result.records.iter().zip(entry.records.iter()).enumerate() {
            let fields: [(f64, f64, &str); 6] = [
                (new.current_smoothed, old.1, "current_smoothed"),
                (new.current_real, old.2, "current_real"),
                (new.basin_smoothed_from_smoothed, old.3, "basin_smoothed_from_smoothed"),
                (new.basin_real_from_smoothed, old.4, "basin_real_from_smoothed"),
                (new.basin_smoothed_from_real, old.5, "basin_smoothed_from_real"),
                (new.basin_real_from_real, old.6, "basin_real_from_real"),
            ];
            assert_eq!(new.step, old.0, "EoFlip step mismatch at record {}", i);
            for &(n, o, name) in &fields {
                let diff = (n - o).abs();
                let tol = 1e-12 * (1.0 + n.abs().max(o.abs()));
                if diff > tol {
                    mismatched_records.push(format!(
                        "tau={}, seed={}, rec={}, field={}, new={}, old={}, diff={}",
                        entry.tau, entry.seed, i, name, n, o, diff
                    ));
                }
            }
        }
        total += 1;
    }

    assert!(
        mismatched_records.is_empty(),
        "EoFlip f64 record mismatches ({} entries):\n{}",
        mismatched_records.len(),
        mismatched_records.join("\n")
    );

    assert_eq!(total, 40, "expected 2 taus × 20 seeds = 40 EoFlip entries");
}
