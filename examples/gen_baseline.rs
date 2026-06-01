//! Regression baseline generator.
//!
//! 現行実装の出力を JSON で吐き出し、回帰テスト (`tests/regression.rs`) の
//! baseline とする。アルゴリズムを意図的に変更したら再実行して更新する。
//!
//! 実行: `cargo run --release --example gen_baseline > tests/data/regression_baseline.json`

use gpp_utils::graph_spec::{GraphKind, GraphSpec, StoredGraph};
use gpp_utils::run_config::{RunConfig, SmoothingSpec, SolverSpec};
use gpp_utils::run_executor::execute;
use rayon::prelude::*;
use serde::Serialize;

#[derive(Serialize)]
struct BaselineEntry {
    smoothing: String,
    seed: u64,
    final_partition: Vec<bool>,
    records: Vec<(usize, f64, f64, f64, f64, f64, f64)>,
}

/// τ-EO の baseline エントリ（SA とは別軸 = τ で識別）。
#[derive(Serialize)]
struct EoEntry {
    tau: f64,
    seed: u64,
    final_partition: Vec<bool>,
    records: Vec<(usize, f64, f64, f64, f64, f64, f64)>,
}

#[derive(Serialize)]
struct Baseline {
    graph_spec: GraphSpec,
    log10_iterations: u32,
    theta: Option<f64>,
    entries: Vec<BaselineEntry>,
    /// EO エントリ（加法的拡張。読み込み側は `#[serde(default)]`）。
    eo_entries: Vec<EoEntry>,
}

fn main() {
    // 検証 5 の小規模設定: N=30, d=4, log10_iter=3 (max=1000), 20 seed × 4 smoothing
    let spec = GraphSpec {
        kind: GraphKind::Random,
        n: 30,
        d: 4.0,
        seed: 0,
    };
    let stored = StoredGraph::generate(spec);
    let prob = stored.problem();

    let smoothings = [
        ("none", SmoothingSpec::None),
        ("kavg8", SmoothingSpec::KAverage(8)),
        ("rkavg8", SmoothingSpec::RandomKAverage(8)),
        ("wavg0.5", SmoothingSpec::WeightedAverage(0.5)),
    ];

    // smoothing × seed をシード並列で実行する。`collect()` は順序を保つため、
    // 出力 JSON は逐次版とバイト単位で一致する。
    let prob = &prob;
    let entries: Vec<BaselineEntry> = smoothings
        .par_iter()
        .flat_map(move |(label, sm)| {
            let label = *label;
            let sm = *sm;
            (0..20u64).into_par_iter().map(move |seed| {
                let mut cfg = RunConfig::new("regression");
                cfg.theta = Some(0.0);
                cfg.log10_iterations = 3;
                cfg.smoothing = sm;
                let r = execute(spec, &cfg, prob, seed);
                let records: Vec<_> = r
                    .records
                    .iter()
                    .map(|sr| {
                        (
                            sr.step,
                            sr.current_smoothed,
                            sr.current_real,
                            sr.basin_smoothed_from_smoothed,
                            sr.basin_real_from_smoothed,
                            sr.basin_smoothed_from_real,
                            sr.basin_real_from_real,
                        )
                    })
                    .collect();
                BaselineEntry {
                    smoothing: label.to_string(),
                    seed,
                    final_partition: r.final_partition,
                    records,
                }
            })
        })
        .collect();

    // EO baseline: τ ∈ {1.2, 1.4} × 20 seed（同じ N=30 グラフ, log10_iter=3）。
    let taus = [1.2_f64, 1.4_f64];
    let eo_entries: Vec<EoEntry> = taus
        .par_iter()
        .flat_map(|&tau| {
            (0..20u64).into_par_iter().map(move |seed| {
                let mut cfg = RunConfig::new("regression-eo");
                cfg.theta = None;
                cfg.smoothing = SmoothingSpec::None;
                cfg.log10_iterations = 3;
                cfg.solver = SolverSpec::Eo { tau };
                let r = execute(spec, &cfg, prob, seed);
                let records: Vec<_> = r
                    .records
                    .iter()
                    .map(|sr| {
                        (
                            sr.step,
                            sr.current_smoothed,
                            sr.current_real,
                            sr.basin_smoothed_from_smoothed,
                            sr.basin_real_from_smoothed,
                            sr.basin_smoothed_from_real,
                            sr.basin_real_from_real,
                        )
                    })
                    .collect();
                EoEntry {
                    tau,
                    seed,
                    final_partition: r.final_partition,
                    records,
                }
            })
        })
        .collect();

    let baseline = Baseline {
        graph_spec: spec,
        log10_iterations: 3,
        theta: Some(0.0),
        entries,
        eo_entries,
    };

    let json = serde_json::to_string_pretty(&baseline).expect("serialize");
    println!("{}", json);
}
