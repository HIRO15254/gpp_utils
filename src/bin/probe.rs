//! 収穫済み状態スナップショットへのオフラインプローブ CLI。
//!
//! `cli --save-states` で収穫したストア（`seed_X_states.json`）を走査し、
//! 全プローブ設定の λ を再計算して設定ペアごとの順位・選択分布距離を
//! 集約 CSV（`pairs_<cond>.csv` / `specs_<cond>.csv`）に書き出す。
//!
//! 使い方:
//!   cargo run --release --bin probe -- --store data/results_rankdiv --out data/rankdiv_probe

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use gpp_utils::probe::{
    default_jaccard_ms, default_probe_specs, default_probe_taus, run_probe, ProbeConfig,
    VertexDump,
};
use gpp_utils::run_config::EoFlipFitnessSpec;

/// EoFlip 状態スナップショットのオフライン順位プローブ。
#[derive(Parser)]
#[command(name = "gpp-probe", version, about)]
struct Args {
    /// 収穫ストア（`cli --out` で指定したディレクトリ）。
    #[arg(long, value_name = "DIR")]
    store: PathBuf,

    /// グラフのロード／生成キャッシュ先ディレクトリ。
    #[arg(long, value_name = "DIR", default_value = "data/graphs")]
    graphs: PathBuf,

    /// 集約 CSV の出力先ディレクトリ。
    #[arg(long, value_name = "DIR", default_value = "data/rankdiv_probe")]
    out: PathBuf,

    /// プローブ設定リストの JSON ファイル（EoFlipFitnessSpec の配列）。省略時は既定 36 設定。
    #[arg(long, value_name = "FILE")]
    specs: Option<PathBuf>,

    /// 選択分布指標を計算する τ（カンマ区切り）。既定 0.8,1.1,1.4,1.7。
    #[arg(long, value_delimiter = ',')]
    taus: Vec<f64>,

    /// bottom-m Jaccard の m（カンマ区切り）。既定 1,2,4,8,16,32,64,128。
    #[arg(long, value_delimiter = ',')]
    jaccard_ms: Vec<usize>,

    /// 対象アルゴ seed（カンマ区切り）。省略時は全 seed。
    #[arg(long, value_delimiter = ',')]
    seeds: Option<Vec<u64>>,

    /// ランダム順位の疑似設定（物差し）を追加する。
    #[arg(long)]
    random_anchor: bool,

    /// 並列ワーカ数（省略時は論理コア数）。
    #[arg(long, value_name = "N")]
    threads: Option<usize>,

    /// 頂点レベルダンプ（vertices.csv）を出すステップ（カンマ区切り）。省略時はダンプなし。
    #[arg(long, value_delimiter = ',')]
    dump_steps: Option<Vec<usize>>,

    /// 頂点ダンプ対象の収穫側設定ラベル（例: mulgamma）。既定 mulgamma。
    #[arg(long, default_value = "mulgamma")]
    dump_src: String,

    /// 頂点ダンプ対象のグラフ instance seed。既定 0。
    #[arg(long, default_value_t = 0)]
    dump_graph_seed: u64,

    /// 頂点ダンプ対象のアルゴ seed。既定 0。
    #[arg(long, default_value_t = 0)]
    dump_seed: u64,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let specs: Vec<EoFlipFitnessSpec> = match &args.specs {
        None => default_probe_specs(),
        Some(p) => match gpp_utils::file_utils::load_json(p) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("プローブ設定の読み込みに失敗: {}: {}", p.display(), e);
                return ExitCode::FAILURE;
            }
        },
    };

    let threads = args
        .threads
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1);

    let cfg = ProbeConfig {
        store_dir: args.store,
        graph_dir: args.graphs,
        out_dir: args.out.clone(),
        specs,
        taus: if args.taus.is_empty() { default_probe_taus() } else { args.taus },
        jaccard_ms: if args.jaccard_ms.is_empty() { default_jaccard_ms() } else { args.jaccard_ms },
        seeds: args.seeds,
        include_random_anchor: args.random_anchor,
        threads,
        vertex_dump: args.dump_steps.map(|steps| VertexDump {
            steps,
            src_label: args.dump_src,
            graph_seed: args.dump_graph_seed,
            seed: args.dump_seed,
        }),
    };

    eprintln!(
        "probe 開始: specs={} taus={:?} ms={:?} threads={}",
        cfg.specs.len() + cfg.include_random_anchor as usize,
        cfg.taus,
        cfg.jaccard_ms,
        threads
    );
    let t0 = std::time::Instant::now();
    match run_probe(&cfg) {
        Ok(s) => {
            eprintln!(
                "完了 ({:.1}s): runs={} states={} groups={} rows: pairs={} specs={} vertices={} errors={}",
                t0.elapsed().as_secs_f64(),
                s.runs,
                s.states,
                s.groups,
                s.rows_pairs,
                s.rows_specs,
                s.rows_vertices,
                s.errors.len()
            );
            for e in s.errors.iter().take(20) {
                eprintln!("  error: {}", e);
            }
            eprintln!("出力: {}", args.out.display());
            if s.errors.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("probe 失敗: {}", e);
            ExitCode::FAILURE
        }
    }
}
