//! GUI / CLI 共通のバッチ実行ランナー。
//!
//! グラフ × 設定 × シードのジョブを rayon で並列実行し、各ジョブを
//! 最適化済み高速パス [`crate::run_executor::execute`] で処理する。
//! GUI の `start_run` と CLI バイナリの双方がこのモジュールを呼ぶことで、
//! 実行パス（高速化）と出力フォーマットを完全に共有する。

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::graph_partition::GraphPartitionProblem;
use crate::graph_spec::{GraphLibrary, GraphSpec};
use crate::run_config::RunConfig;
use crate::run_executor::{ResultStore, execute};

/// バッチ実行の指定。CLI ではこれを JSON ファイルとして受け取る。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSpec {
    /// 実行対象グラフの仕様（未生成なら自動生成・キャッシュされる）。
    pub graphs: Vec<GraphSpec>,
    /// 実行する SA 設定。
    pub configs: Vec<RunConfig>,
    /// シードの開始値。
    pub seed_start: u64,
    /// シード本数（`seed_start, seed_start+1, ...` を `seed_count` 個実行）。
    pub seed_count: usize,
}

impl BatchSpec {
    /// 総ジョブ数（graphs × configs × seeds）。
    pub fn total_jobs(&self) -> usize {
        self.graphs.len() * self.configs.len() * self.seed_count
    }
}

/// バッチ実行中に発火する進捗イベント。GUI / CLI が各自で整形・集計する。
#[derive(Debug, Clone)]
pub enum BatchEvent {
    /// 実行開始（最初に 1 度だけ）。
    Started {
        total: usize,
        graphs: usize,
        configs: usize,
        seeds: usize,
        threads: usize,
    },
    /// 既存結果があったためスキップした。
    Skipped {
        graph: String,
        config: String,
        seed: u64,
    },
    /// 1 ジョブが完了し結果を保存した。
    Done {
        graph: String,
        config: String,
        seed: u64,
        elapsed_s: f64,
        final_real: f64,
    },
    /// 結果保存に失敗した。
    SaveError { message: String },
    /// rayon スレッドプールの構築に失敗した。
    PoolError { message: String },
    /// グラフのロード／生成に失敗した（その仕様のジョブは実行されない）。
    GraphError { spec: GraphSpec, message: String },
    /// 実行終了（最後に必ず 1 度発火）。
    Finished,
}

/// 1 並列ワーカが消化する実行ジョブ。
struct Job {
    graph_spec: GraphSpec,
    /// グラフ単位で 1 つ作って Arc 共有する（毎ジョブ作り直さない）。
    problem: Arc<GraphPartitionProblem>,
    config: RunConfig,
    seed: u64,
}

/// バッチを同期実行する。呼び出し側がスレッド管理を行う前提
/// （GUI は裏スレッドから、CLI はメインスレッドから呼ぶ）。
///
/// - `graph_dir`: グラフのロード／生成キャッシュ先（例: `data/graphs`）。
/// - `store_dir`: 結果 JSON の保存先（例: `data/results`）。
/// - `skip_existing`: true なら既存結果のあるジョブをスキップする。
/// - `cancel`: true がセットされると未処理ジョブを実行せず素通りする。
/// - `on_event`: 進捗コールバック。複数ワーカから並行に呼ばれるため `Sync`。
pub fn run_batch<F>(
    spec: &BatchSpec,
    graph_dir: &Path,
    store_dir: &Path,
    num_threads: usize,
    skip_existing: bool,
    cancel: Arc<AtomicBool>,
    on_event: F,
) where
    F: Fn(BatchEvent) + Sync,
{
    let num_threads = num_threads.max(1);
    on_event(BatchEvent::Started {
        total: spec.total_jobs(),
        graphs: spec.graphs.len(),
        configs: spec.configs.len(),
        seeds: spec.seed_count,
        threads: num_threads,
    });

    // グラフごとに Problem を 1 度だけ構築し Arc 共有する。
    let library = GraphLibrary::new(graph_dir);
    let mut jobs: Vec<Job> = Vec::with_capacity(spec.total_jobs());
    for &gspec in &spec.graphs {
        let stored = match library.load_or_generate(gspec) {
            Ok(g) => g,
            Err(e) => {
                on_event(BatchEvent::GraphError { spec: gspec, message: e });
                continue;
            }
        };
        let problem = Arc::new(stored.problem());
        for cfg in &spec.configs {
            for s_off in 0..spec.seed_count {
                let seed = spec.seed_start.wrapping_add(s_off as u64);
                jobs.push(Job {
                    graph_spec: gspec,
                    problem: Arc::clone(&problem),
                    config: cfg.clone(),
                    seed,
                });
            }
        }
    }

    let store = ResultStore::new(store_dir);

    // rayon のローカルスレッドプールでジョブを並列消化する。
    match rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
    {
        Ok(pool) => pool.install(|| {
            jobs.into_par_iter().for_each(|job| {
                // キャンセル済みなら即 return（残りジョブも素通りする）。
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                if skip_existing && store.exists(&job.graph_spec, &job.config, job.seed) {
                    on_event(BatchEvent::Skipped {
                        graph: job.graph_spec.id(),
                        config: job.config.id(),
                        seed: job.seed,
                    });
                    return;
                }
                let t0 = std::time::Instant::now();
                let result = execute(job.graph_spec, &job.config, &job.problem, job.seed);
                let elapsed_s = t0.elapsed().as_secs_f64();
                if let Err(e) = store.save(&result) {
                    on_event(BatchEvent::SaveError { message: e });
                }
                on_event(BatchEvent::Done {
                    graph: job.graph_spec.id(),
                    config: job.config.id(),
                    seed: job.seed,
                    elapsed_s,
                    final_real: result
                        .records
                        .last()
                        .map(|r| r.current_real)
                        .unwrap_or(f64::NAN),
                });
            });
        }),
        Err(e) => on_event(BatchEvent::PoolError {
            message: e.to_string(),
        }),
    }

    on_event(BatchEvent::Finished);
}
