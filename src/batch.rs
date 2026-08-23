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
use crate::run_config::{ConfigSweep, RunConfig};
use crate::run_executor::{ResultStore, execute_with_states};

/// バッチ実行の指定。CLI ではこれを JSON ファイルとして受け取る。
///
/// 実行される設定は「明示列挙した `configs`」と「`config_sweep` を直積展開したもの」
/// を連結したもの（[`BatchSpec::effective_configs`]）。どちらか一方だけでもよい。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSpec {
    /// 実行対象グラフの仕様（未生成なら自動生成・キャッシュされる）。
    pub graphs: Vec<GraphSpec>,
    /// 明示的に列挙した SA 設定。
    #[serde(default)]
    pub configs: Vec<RunConfig>,
    /// 温度×反復回数×平滑化を総当たり展開する指定（任意）。
    #[serde(default)]
    pub config_sweep: Option<ConfigSweep>,
    /// シードの開始値。
    pub seed_start: u64,
    /// シード本数（`seed_start, seed_start+1, ...` を `seed_count` 個実行）。
    pub seed_count: usize,
    /// EoFlip 系ランで分割スナップショットを `seed_X_states.json` に保存する（既定 false）。
    /// 非 EoFlip ソルバーには影響しない。true のとき skip 判定は states ファイルの存在も要求する。
    #[serde(default)]
    pub save_states: bool,
}

impl BatchSpec {
    /// 実際に実行する設定一覧（明示列挙 + sweep 展開）。
    pub fn effective_configs(&self) -> Vec<RunConfig> {
        let mut cfgs = self.configs.clone();
        if let Some(sweep) = &self.config_sweep {
            cfgs.extend(sweep.expand());
        }
        cfgs
    }

    /// 総ジョブ数（graphs × effective_configs × seeds）。
    pub fn total_jobs(&self) -> usize {
        self.graphs.len() * self.effective_configs().len() * self.seed_count
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
    // 明示設定 + sweep 展開を 1 度だけ確定させる。
    let configs = spec.effective_configs();
    let total = spec.graphs.len() * configs.len() * spec.seed_count;
    on_event(BatchEvent::Started {
        total,
        graphs: spec.graphs.len(),
        configs: configs.len(),
        seeds: spec.seed_count,
        threads: num_threads,
    });

    // グラフごとに Problem を 1 度だけ構築し Arc 共有する。
    let library = GraphLibrary::new(graph_dir);
    let mut jobs: Vec<Job> = Vec::with_capacity(total);
    for &gspec in &spec.graphs {
        let stored = match library.load_or_generate(gspec) {
            Ok(g) => g,
            Err(e) => {
                on_event(BatchEvent::GraphError { spec: gspec, message: e });
                continue;
            }
        };
        let problem = Arc::new(stored.problem());
        for cfg in &configs {
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
                // save_states 時は EoFlip 系にだけ states ファイルの存在も要求する
                // （非 EoFlip は states を作らないので結果のみで skip 可能）。
                let wants_states = spec.save_states
                    && crate::run_config::EoFlipFitnessSpec::from_solver(&job.config.solver)
                        .is_some();
                if skip_existing
                    && store.exists(&job.graph_spec, &job.config, job.seed)
                    && (!wants_states
                        || store.states_exist(&job.graph_spec, &job.config, job.seed))
                {
                    on_event(BatchEvent::Skipped {
                        graph: job.graph_spec.id(),
                        config: job.config.id(),
                        seed: job.seed,
                    });
                    return;
                }
                let t0 = std::time::Instant::now();
                let (result, states) = execute_with_states(
                    job.graph_spec,
                    &job.config,
                    &job.problem,
                    job.seed,
                    spec.save_states,
                );
                let elapsed_s = t0.elapsed().as_secs_f64();
                if let Err(e) = store.save(&result) {
                    on_event(BatchEvent::SaveError { message: e });
                }
                if let Some(st) = states {
                    if let Err(e) = store.save_states(&st) {
                        on_event(BatchEvent::SaveError { message: e });
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `flip_trace` フィールドを持つ旧バッチ JSON は未知フィールドとして無視され、
    /// 読み込み自体は失敗しない（後方互換）。
    #[test]
    fn test_legacy_flip_trace_field_is_ignored() {
        let json = r#"{
            "graphs": [{"kind": "Random", "n": 20, "d": 3.0, "seed": 0}],
            "configs": [],
            "seed_start": 0,
            "seed_count": 1,
            "flip_trace": {"probes": [
                {"Legacy": {"alpha_eo": 0.064, "diff_exp": 2.0}},
                "MulGamma"
            ]}
        }"#;
        let spec: BatchSpec = serde_json::from_str(json).expect("deserialize");
        assert_eq!(spec.effective_configs().len(), 0);
    }

    /// `save_states` を持たない既存バッチ JSON は false として読み込まれる（後方互換）。
    #[test]
    fn test_save_states_defaults_false() {
        let json = r#"{
            "graphs": [{"kind": "Random", "n": 20, "d": 3.0, "seed": 0}],
            "configs": [],
            "seed_start": 0,
            "seed_count": 1
        }"#;
        let spec: BatchSpec = serde_json::from_str(json).expect("deserialize");
        assert!(!spec.save_states);
    }

    /// 旧バッチ JSON に残る `eo_tie_break` は未知フィールドとして無視され、
    /// 展開後の id にも影響しない（tie 規則が 1 本化されたため）。
    #[test]
    fn test_legacy_eo_tie_break_field_is_ignored() {
        let json = r#"{
            "graphs": [{"kind": "Random", "n": 20, "d": 3.0, "seed": 0}],
            "config_sweep": {
                "log10_iterations": [4],
                "solver_kind": "EoFlipMulGamma",
                "taus": [1.4]
            },
            "seed_start": 0,
            "seed_count": 1,
            "eo_tie_break": "CompetitionRandom"
        }"#;
        let spec: BatchSpec = serde_json::from_str(json).expect("deserialize");
        let cfgs = spec.effective_configs();
        assert_eq!(cfgs.len(), 1);
        assert_eq!(cfgs[0].id(), "eoflipmulgamma_iter4_tau1p4");

        // 旧 "Index" 指定でも同じ id になる。
        let json_idx = json.replace("CompetitionRandom", "Index");
        let spec_idx: BatchSpec = serde_json::from_str(&json_idx).expect("deserialize");
        assert_eq!(spec_idx.effective_configs()[0].id(), "eoflipmulgamma_iter4_tau1p4");
    }
}
