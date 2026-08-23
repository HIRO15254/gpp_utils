//! 収穫（save_states）→ オフラインプローブの一気通貫テスト。
//!
//! 小さなバッチを `run_batch` の save_states 有効で実行して states ファイルを作り、
//! `probe::run_probe` を掛けて CSV の行数・値域を検証する。

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use gpp_utils::batch::{run_batch, BatchSpec};
use gpp_utils::graph_spec::{GraphKind, GraphSpec};
use gpp_utils::probe::{run_probe, ProbeConfig, VertexDump, BAND_NAMES};
use gpp_utils::run_config::{EoFlipFitnessSpec, RunConfig, SmoothingSpec, SolverSpec};

fn eoflip_cfg(name: &str, solver: SolverSpec) -> RunConfig {
    let mut cfg = RunConfig::new(name);
    cfg.log10_iterations = 2;
    cfg.smoothing = SmoothingSpec::None;
    cfg.solver = solver;
    cfg
}

#[test]
fn test_harvest_then_probe_end_to_end() {
    let base = std::env::temp_dir().join(format!("gpp_probe_it_{}", std::process::id()));
    let graph_dir = base.join("graphs");
    let store_dir = base.join("results");
    let out_dir = base.join("probe");
    let _ = std::fs::remove_dir_all(&base);

    let gspec = GraphSpec { kind: GraphKind::Random, n: 30, d: 4.0, seed: 0 };
    let spec = BatchSpec {
        graphs: vec![gspec],
        configs: vec![
            eoflip_cfg("mg", SolverSpec::EoFlipMulGamma { tau: 1.4 }),
            eoflip_cfg("ab", SolverSpec::EoFlipAddBeta { tau: 1.4, beta: 0.0 }),
        ],
        config_sweep: None,
        seed_start: 0,
        seed_count: 2,
        save_states: true,
    };
    let cancel = Arc::new(AtomicBool::new(false));
    run_batch(&spec, &graph_dir, &store_dir, 2, true, Arc::clone(&cancel), |_| {});

    // states ファイルが4ラン分あること。
    let mut states_count = 0;
    for cfg in &spec.configs {
        for seed in 0..2u64 {
            let p = store_dir
                .join(gspec.id())
                .join(cfg.id())
                .join(format!("seed_{}_states.json", seed));
            assert!(p.exists(), "states がない: {}", p.display());
            states_count += 1;
        }
    }
    assert_eq!(states_count, 4);

    // 再実行すると skip される（結果 + states 両方あるので再計算されない）。
    let mut skipped = std::sync::atomic::AtomicUsize::new(0);
    run_batch(&spec, &graph_dir, &store_dir, 2, true, cancel, |ev| {
        if matches!(ev, gpp_utils::batch::BatchEvent::Skipped { .. }) {
            skipped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });
    assert_eq!(*skipped.get_mut(), 4, "冪等性: 全ジョブ skip のはず");

    // プローブ実行（3 設定 + anchor、2 τ、m=2）。
    let probe_cfg = ProbeConfig {
        store_dir: store_dir.clone(),
        graph_dir: graph_dir.clone(),
        out_dir: out_dir.clone(),
        specs: vec![
            EoFlipFitnessSpec::MulGamma,
            EoFlipFitnessSpec::AddBeta { beta: 0.0 },
            EoFlipFitnessSpec::MulAlpha { alpha: 1.0 },
        ],
        taus: vec![0.8, 1.4],
        jaccard_ms: vec![2],
        seeds: None,
        include_random_anchor: true,
        threads: 2,
        vertex_dump: Some(VertexDump {
            steps: vec![0, 100],
            src_label: "mulgamma".to_string(),
            graph_seed: 0,
            seed: 0,
        }),
    };
    let summary = run_probe(&probe_cfg).expect("probe 成功");
    assert_eq!(summary.runs, 4);
    // iter=10^2 → snapshots = 1 (step0) + logarithmic_steps(100).len()
    // logarithmic_steps(100) = 1..9, 10..90, 100 → 19 個 → 20 snapshots/run。
    assert_eq!(summary.states, 4 * 20);
    assert!(summary.errors.is_empty(), "errors: {:?}", summary.errors);

    // pairs CSV: グループ = (cond, src, tau) は 2 つ（mg と ab、cond/τ 共通）。
    // 4 設定 → 6 ペア。iter100 では band は init/early のみ（step<=100 のうち 100 は early）。
    let pairs_csv =
        std::fs::read_to_string(out_dir.join("pairs_random_n30_d4.csv")).expect("pairs csv");
    let lines: Vec<&str> = pairs_csv.lines().collect();
    assert!(lines[0].starts_with("cond,src_setting,src_tau,band,"));
    let data_rows = &lines[1..];
    // 2 グループ × 2 帯 (init, early) × 6 ペア = 24 行。
    assert_eq!(data_rows.len(), 2 * 2 * 6, "rows: {}", data_rows.len());
    for row in data_rows {
        let cols: Vec<&str> = row.split(',').collect();
        assert_eq!(cols[0], "random_n30_d4");
        assert!(BAND_NAMES.contains(&cols[3]));
        // kendall_b_mean ∈ [-1,1]（NaN 許容: 縮退時）。
        let tb: f64 = cols[8].parse().unwrap();
        assert!(tb.is_nan() || (-1.0..=1.0).contains(&tb), "tau_b={tb}");
        // tv/jsd/jacc ∈ [0,1]。
        for c in &cols[11..] {
            let v: f64 = c.parse().unwrap();
            assert!((0.0..=1.0 + 1e-9).contains(&v), "metric={v}");
        }
    }

    // specs CSV。
    let specs_csv =
        std::fs::read_to_string(out_dir.join("specs_random_n30_d4.csv")).expect("specs csv");
    // 2 グループ × 2 帯 × 4 設定 = 16 データ行。
    assert_eq!(specs_csv.lines().count() - 1, 16);

    // vertices CSV: mulgamma / graph_seed 0 / seed 0 のランのみ、2 step × 30 頂点 × 4 設定。
    let vertices_csv = std::fs::read_to_string(out_dir.join("vertices.csv")).expect("vertices");
    assert_eq!(vertices_csv.lines().count() - 1, 2 * 30 * 4);

    // manifest。
    assert!(out_dir.join("manifest.json").exists());

    let _ = std::fs::remove_dir_all(&base);
}
