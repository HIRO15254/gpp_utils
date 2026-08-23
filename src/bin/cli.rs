//! ヘッドレスなバッチ実行 CLI。
//!
//! GUI (`gui` バイナリ) はグラフィックアダプタを要求するため GPU 非搭載の
//! Azure VM 等では起動できない。この CLI は同じ最適化済み実行パス
//! (`gpp_utils::batch::run_batch` → `run_executor::execute`) を GPU 非依存で
//! 呼び出し、結果を GUI と同一レイアウト (`data/results/<graph>/<config>/seed_*.json`)
//! で保存する。保存された結果はローカルの GUI Results タブでそのまま閲覧できる。
//!
//! 使い方:
//!   cargo run --release --bin cli -- --batch examples/batch.example.json

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use clap::Parser;

use gpp_utils::batch::{BatchEvent, BatchSpec, run_batch};
use gpp_utils::file_utils::load_json;

/// グラフ分割最適化のバッチ実行（ヘッドレス・GPU 不要）。
#[derive(Parser)]
#[command(name = "gpp-cli", version, about)]
struct Args {
    /// 実行内容を記述した JSON バッチ定義ファイル。
    #[arg(long, value_name = "FILE")]
    batch: PathBuf,

    /// 結果 JSON の保存先ディレクトリ。
    #[arg(long, value_name = "DIR", default_value = "data/results")]
    out: PathBuf,

    /// グラフのロード／生成キャッシュ先ディレクトリ。
    #[arg(long, value_name = "DIR", default_value = "data/graphs")]
    graphs: PathBuf,

    /// 並列ワーカ数（省略時は論理コア数）。
    #[arg(long, value_name = "N")]
    threads: Option<usize>,

    /// 既存結果も上書き再計算する（既定は既存結果をスキップ）。
    #[arg(long)]
    overwrite: bool,

    /// EoFlip 系ランで分割スナップショット（seed_X_states.json）も保存する
    /// （バッチ定義の save_states と OR）。
    #[arg(long)]
    save_states: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let mut spec: BatchSpec = match load_json(&args.batch) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "バッチ定義の読み込みに失敗しました: {}: {}",
                args.batch.display(),
                e
            );
            return ExitCode::FAILURE;
        }
    };
    spec.save_states |= args.save_states;

    let total = spec.total_jobs();
    if total == 0 {
        eprintln!(
            "実行対象がありません。graphs / configs / seed_count を確認してください。"
        );
        return ExitCode::FAILURE;
    }

    let threads = args
        .threads
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1)
        .max(1);

    // 進捗カウンタ（複数ワーカから更新されるため atomic）。
    let done = AtomicUsize::new(0);
    let errors = AtomicUsize::new(0);
    // CLI ではキャンセル UI を持たないが、API 上必要なので未セットのまま渡す。
    let cancel = Arc::new(AtomicBool::new(false));

    run_batch(
        &spec,
        &args.graphs,
        &args.out,
        threads,
        !args.overwrite,
        Arc::clone(&cancel),
        |ev| match ev {
            BatchEvent::Started { total, graphs, configs, seeds, threads } => {
                eprintln!(
                    "開始: {} jobs ({} graphs x {} configs x {} seeds, {} threads)",
                    total, graphs, configs, seeds, threads
                );
            }
            BatchEvent::Skipped { graph, config, seed } => {
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!("[{}/{}] skip {} / {} / seed={}", n, total, graph, config, seed);
            }
            BatchEvent::Done { graph, config, seed, elapsed_s, final_real } => {
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!(
                    "[{}/{}] done {} / {} / seed={} ({:.1}s, real={:.2})",
                    n, total, graph, config, seed, elapsed_s, final_real
                );
            }
            BatchEvent::SaveError { message } => {
                errors.fetch_add(1, Ordering::Relaxed);
                eprintln!("  保存エラー: {}", message);
            }
            BatchEvent::PoolError { message } => {
                errors.fetch_add(1, Ordering::Relaxed);
                eprintln!("  スレッドプールエラー: {}", message);
            }
            BatchEvent::GraphError { spec, message } => {
                errors.fetch_add(1, Ordering::Relaxed);
                eprintln!("  グラフエラー {}: {}", spec.id(), message);
            }
            BatchEvent::Finished => {
                eprintln!("完了。結果は {} 以下に保存されました。", args.out.display());
            }
        },
    );

    if errors.load(Ordering::Relaxed) > 0 {
        eprintln!("{} 件のエラーが発生しました。", errors.load(Ordering::Relaxed));
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
