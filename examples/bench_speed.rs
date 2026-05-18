//! 速度ベンチマーク。
//!
//! 設定: N=500, 期待次数 5, 10^4 iter, 4 種の smoothing × 5 seed の平均 elapsed_ms。
//! 旧/新の elapsed_ms 比較に使用する。
//!
//! 実行: `cargo run --release --example bench_speed`

use gpp_utils::graph_spec::{GraphKind, GraphSpec, StoredGraph};
use gpp_utils::run_config::{RunConfig, SmoothingSpec};
use gpp_utils::run_executor::execute;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn partition_hash(p: &[bool]) -> u64 {
    let mut h = DefaultHasher::new();
    p.hash(&mut h);
    h.finish()
}

fn main() {
    let spec = GraphSpec {
        kind: GraphKind::Random,
        n: 500,
        d: 5.0,
        seed: 0,
    };
    let stored = StoredGraph::generate(spec);
    let prob = stored.problem();

    let smoothings: Vec<(&str, SmoothingSpec)> = vec![
        ("None", SmoothingSpec::None),
        ("KAvg(8)", SmoothingSpec::KAverage(8)),
        ("RandomK(8)", SmoothingSpec::RandomKAverage(8)),
        ("Weighted(0.5)", SmoothingSpec::WeightedAverage(0.5)),
    ];

    println!("{}", "=".repeat(72));
    println!(
        "Benchmark: N={}, d={}, max_iter=10^4, 5 seeds per config",
        spec.n, spec.d
    );
    println!("Edges in graph: {}", stored.edge_count);
    println!("{}", "=".repeat(72));
    println!(
        "{:<15} | {:>12} | {:>12} | {:>12} | {:<20}",
        "Smoothing", "min_ms", "avg_ms", "max_ms", "final_partition_hash"
    );
    println!("{}", "-".repeat(72));

    for (label, sm) in &smoothings {
        let mut cfg = RunConfig::new("bench");
        cfg.theta = Some(0.0);
        cfg.log10_iterations = 4;
        cfg.smoothing = *sm;

        let mut times = Vec::new();
        let mut hashes = Vec::new();
        for seed in 0..5u64 {
            let r = execute(spec, &cfg, &prob, seed);
            times.push(r.elapsed_ms);
            hashes.push(partition_hash(&r.final_partition));
        }

        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg: f64 = times.iter().sum::<f64>() / times.len() as f64;

        // hashes を 16 進で前 8 桁
        let hash_summary: String = hashes
            .iter()
            .map(|h| format!("{:08x}", h & 0xFFFF_FFFF))
            .collect::<Vec<_>>()
            .join(",");

        println!(
            "{:<15} | {:>10.2}ms | {:>10.2}ms | {:>10.2}ms | {}",
            label, min, avg, max, hash_summary
        );
    }
    println!("{}", "=".repeat(72));
}
