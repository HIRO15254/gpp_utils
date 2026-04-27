//! 対数刻みでスナップショットを取りながら SA を実行する。
//!
//! 各スナップショットでは現在解と、スムージング空間および元空間それぞれで
//! 山登りを行ったベイスンの評価値を記録する。

use std::path::{Path, PathBuf};

use rand::Rng;
use rand_mt::Mt19937GenRand64;
use serde::{Deserialize, Serialize};

use crate::file_utils::{ensure_dir_exists, load_json, save_json};
use crate::graph_partition::{GraphPartitionProblem, Partition, get_partition_sizes};
use crate::graph_spec::GraphSpec;
use crate::optimization::{Problem, Smoothing};
use crate::run_config::{RunConfig, SmoothingSpec};
use crate::smoothing::{
    KAveragingSmoothing, RandomKSmoothing, WeightedNeighbourSmoothing,
};

/// 1 ステップ分の計測値。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// SA のステップ数（0 = 初期解、その後対数刻み）。
    pub step: usize,
    /// 現在解の現在（スムージング）空間での評価値。
    pub current_smoothed: f64,
    /// 現在解の元空間での評価値（=実スコア）。
    pub current_real: f64,
    /// 現在空間で山登り → そのベイスンの現在空間評価値。
    pub basin_smoothed_from_smoothed: f64,
    /// 現在空間で山登り → そのベイスンの元空間評価値。
    pub basin_real_from_smoothed: f64,
    /// 元空間で山登り → そのベイスンの現在空間評価値。
    pub basin_smoothed_from_real: f64,
    /// 元空間で山登り → そのベイスンの元空間評価値。
    pub basin_real_from_real: f64,
}

/// 1 シードあたりの実行結果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub graph_spec: GraphSpec,
    pub config: RunConfig,
    pub seed: u64,
    pub final_partition: Partition,
    pub records: Vec<StepRecord>,
    pub elapsed_ms: f64,
}

/// 対数刻みのステップを返す: 1, 2, ..., 9, 10, 20, ..., 90, 100, ...
pub fn logarithmic_steps(max_iter: usize) -> Vec<usize> {
    let mut v = Vec::new();
    if max_iter == 0 {
        return v;
    }
    let mut decade: usize = 1;
    while decade <= max_iter {
        for k in 1..=9 {
            let s = k * decade;
            if s > max_iter {
                break;
            }
            v.push(s);
        }
        if let Some(next) = decade.checked_mul(10) {
            decade = next;
        } else {
            break;
        }
    }
    if v.last().copied() != Some(max_iter) {
        v.push(max_iter);
    }
    v
}

/// 元空間での山登り（差分計算で全近傍を評価し、最良を選ぶ）。
fn hill_climb_real(prob: &GraphPartitionProblem, start: &Partition) -> Partition {
    let mut current = start.clone();
    let n = prob.graph().node_count;
    let (mut t, mut f) = get_partition_sizes(&current);
    loop {
        let mut best_delta = 0.0;
        let mut best_i: Option<usize> = None;
        for i in 0..n {
            let d = prob.flip_delta_with_sizes(&current, i, (t, f));
            if d < best_delta - 1e-12 {
                best_delta = d;
                best_i = Some(i);
            }
        }
        match best_i {
            Some(i) => {
                if current[i] {
                    t -= 1;
                    f += 1;
                } else {
                    t += 1;
                    f -= 1;
                }
                current[i] = !current[i];
            }
            None => break,
        }
    }
    current
}

/// 任意のスムージング空間での山登り（フリップ・イン・プレースで近傍配列確保を回避）。
fn hill_climb_smoothed<Sm>(
    prob: &GraphPartitionProblem,
    sm: &Sm,
    start: &Partition,
) -> Partition
where
    Sm: Smoothing<Partition> + ?Sized,
{
    let mut current = start.clone();
    let n = current.len();
    let mut current_smoothed = sm.score(prob, &current);
    loop {
        let mut best_score = current_smoothed;
        let mut best_i: Option<usize> = None;
        for i in 0..n {
            current[i] = !current[i];
            let s = sm.score(prob, &current);
            current[i] = !current[i];
            if s < best_score {
                best_score = s;
                best_i = Some(i);
            }
        }
        match best_i {
            Some(i) => {
                current[i] = !current[i];
                current_smoothed = best_score;
            }
            None => break,
        }
    }
    current
}

/// SA を実行する（一般スムージング版。`flip_in_place` で近傍配列確保を回避）。
fn run_sa_with_smoothing<Sm>(
    prob: &GraphPartitionProblem,
    sm: &Sm,
    cfg: &RunConfig,
    seed: u64,
) -> (Partition, Vec<StepRecord>)
where
    Sm: Smoothing<Partition> + ?Sized,
{
    let mut rng = Mt19937GenRand64::new(seed);
    let mut current: Partition = prob.random_solution(&mut rng);
    let n = current.len();
    let mut current_smoothed = sm.score(prob, &current);

    let max_iter = cfg.iterations();
    let temperature = cfg.temperature();

    let snap_steps = logarithmic_steps(max_iter);
    let mut snap_iter = snap_steps.iter().copied().peekable();
    let mut records = Vec::with_capacity(snap_steps.len() + 1);

    records.push(make_snapshot(prob, sm, &current, current_smoothed, 0));

    if n == 0 {
        return (current, records);
    }

    for it in 1..=max_iter {
        let i = rng.gen_range(0..n);
        current[i] = !current[i];
        let candidate_smoothed = sm.score(prob, &current);
        let delta = candidate_smoothed - current_smoothed;
        let accept = if delta < 0.0 {
            true
        } else if temperature > 0.0 {
            rng.r#gen::<f64>() < (-delta / temperature).exp()
        } else {
            false
        };
        if accept {
            current_smoothed = candidate_smoothed;
        } else {
            current[i] = !current[i];
        }

        if let Some(&want) = snap_iter.peek() {
            if it == want {
                records.push(make_snapshot(prob, sm, &current, current_smoothed, it));
                snap_iter.next();
            }
        }
    }

    (current, records)
}

/// SA の高速パス（NoSmoothing 専用）。
/// 差分計算でスコアを増分管理し、毎ステップ O(deg(i)) で動く。
fn run_sa_no_smoothing(
    prob: &GraphPartitionProblem,
    cfg: &RunConfig,
    seed: u64,
) -> (Partition, Vec<StepRecord>) {
    let mut rng = Mt19937GenRand64::new(seed);
    let mut current: Partition = prob.random_solution(&mut rng);
    let n = current.len();
    let (mut t, mut f) = get_partition_sizes(&current);
    let mut current_score = prob.score(&current);

    let max_iter = cfg.iterations();
    let temperature = cfg.temperature();

    let snap_steps = logarithmic_steps(max_iter);
    let mut snap_iter = snap_steps.iter().copied().peekable();
    let mut records = Vec::with_capacity(snap_steps.len() + 1);

    records.push(make_snapshot_no_smoothing(prob, &current, current_score, 0));

    if n == 0 {
        return (current, records);
    }

    for it in 1..=max_iter {
        let i = rng.gen_range(0..n);
        let delta = prob.flip_delta_with_sizes(&current, i, (t, f));
        let accept = if delta < 0.0 {
            true
        } else if temperature > 0.0 {
            rng.r#gen::<f64>() < (-delta / temperature).exp()
        } else {
            false
        };
        if accept {
            if current[i] {
                t -= 1;
                f += 1;
            } else {
                t += 1;
                f -= 1;
            }
            current[i] = !current[i];
            current_score += delta;
        }

        if let Some(&want) = snap_iter.peek() {
            if it == want {
                records.push(make_snapshot_no_smoothing(prob, &current, current_score, it));
                snap_iter.next();
            }
        }
    }

    (current, records)
}

/// NoSmoothing 用スナップショット。スムージング空間 = 元空間なので、
/// 6 トレース全てが同じ値になり、山登りも 1 回で済む。
fn make_snapshot_no_smoothing(
    prob: &GraphPartitionProblem,
    current: &Partition,
    current_score: f64,
    step: usize,
) -> StepRecord {
    let basin = hill_climb_real(prob, current);
    let basin_score = prob.score(&basin);
    StepRecord {
        step,
        current_smoothed: current_score,
        current_real: current_score,
        basin_smoothed_from_smoothed: basin_score,
        basin_real_from_smoothed: basin_score,
        basin_smoothed_from_real: basin_score,
        basin_real_from_real: basin_score,
    }
}

fn make_snapshot<Sm>(
    prob: &GraphPartitionProblem,
    sm: &Sm,
    current: &Partition,
    current_smoothed: f64,
    step: usize,
) -> StepRecord
where
    Sm: Smoothing<Partition> + ?Sized,
{
    let current_real = prob.score(current);

    // スムージング空間から山登り
    let basin_smoothed_pt = hill_climb_smoothed(prob, sm, current);
    let basin_smoothed_from_smoothed = sm.score(prob, &basin_smoothed_pt);
    let basin_real_from_smoothed = prob.score(&basin_smoothed_pt);

    // 元空間から山登り
    let basin_real_pt = hill_climb_real(prob, current);
    let basin_smoothed_from_real = sm.score(prob, &basin_real_pt);
    let basin_real_from_real = prob.score(&basin_real_pt);

    StepRecord {
        step,
        current_smoothed,
        current_real,
        basin_smoothed_from_smoothed,
        basin_real_from_smoothed,
        basin_smoothed_from_real,
        basin_real_from_real,
    }
}

/// 単一シードの実行を行い、結果を返す（保存はしない）。
pub fn execute(
    spec: GraphSpec,
    cfg: &RunConfig,
    prob: &GraphPartitionProblem,
    seed: u64,
) -> RunResult {
    let t0 = std::time::Instant::now();
    let sm_seed = seed.wrapping_add(0xDEAD_BEEF);
    let (final_p, records) = match cfg.smoothing {
        // 差分計算を使う高速パス。スムージング空間 = 元空間なので
        // 6 トレースは全て同値、山登りも 1 回で済む。
        SmoothingSpec::None => run_sa_no_smoothing(prob, cfg, seed),
        SmoothingSpec::KAverage(k) => {
            run_sa_with_smoothing(prob, &KAveragingSmoothing::new(k), cfg, seed)
        }
        SmoothingSpec::RandomKAverage(k) => {
            run_sa_with_smoothing(prob, &RandomKSmoothing::new(k, sm_seed), cfg, seed)
        }
        SmoothingSpec::WeightedAverage(k) => {
            run_sa_with_smoothing(prob, &WeightedNeighbourSmoothing::new(k), cfg, seed)
        }
    };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    RunResult {
        graph_spec: spec,
        config: cfg.clone(),
        seed,
        final_partition: final_p,
        records,
        elapsed_ms,
    }
}

/// 結果ストアの管理。
pub struct ResultStore {
    pub base_dir: PathBuf,
}

impl ResultStore {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
        }
    }

    /// 結果ファイルのパス（`base/<graph_id>/<config_id>/seed_<seed>.json`）。
    pub fn path_for(&self, spec: &GraphSpec, cfg: &RunConfig, seed: u64) -> PathBuf {
        self.base_dir
            .join(spec.id())
            .join(cfg.id())
            .join(format!("seed_{}.json", seed))
    }

    pub fn exists(&self, spec: &GraphSpec, cfg: &RunConfig, seed: u64) -> bool {
        self.path_for(spec, cfg, seed).exists()
    }

    pub fn load(&self, spec: &GraphSpec, cfg: &RunConfig, seed: u64) -> Option<RunResult> {
        load_json::<RunResult>(&self.path_for(spec, cfg, seed)).ok()
    }

    pub fn save(&self, result: &RunResult) -> Result<(), String> {
        let p = self.path_for(&result.graph_spec, &result.config, result.seed);
        if let Some(parent) = p.parent() {
            ensure_dir_exists(parent).map_err(|e| format!("create dir: {}", e))?;
        }
        save_json(result, &p).map_err(|e| format!("save: {}", e))
    }

    /// gnuplot で扱いやすい TSV を出力する。
    /// 列: step, current_smoothed, current_real,
    ///     basin_smoothed_from_smoothed, basin_real_from_smoothed,
    ///     basin_smoothed_from_real, basin_real_from_real
    pub fn export_tsv(&self, result: &RunResult, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            ensure_dir_exists(parent).map_err(|e| format!("create dir: {}", e))?;
        }
        use std::io::Write;
        let mut f = std::fs::File::create(path).map_err(|e| format!("create: {}", e))?;
        writeln!(
            f,
            "# graph={} config={} seed={}",
            result.graph_spec.id(),
            result.config.id(),
            result.seed
        )
        .map_err(|e| format!("write: {}", e))?;
        writeln!(
            f,
            "# step\tcur_sm\tcur_real\tbasin_sm_from_sm\tbasin_real_from_sm\tbasin_sm_from_real\tbasin_real_from_real"
        )
        .map_err(|e| format!("write: {}", e))?;
        for r in &result.records {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                r.step,
                r.current_smoothed,
                r.current_real,
                r.basin_smoothed_from_smoothed,
                r.basin_real_from_smoothed,
                r.basin_smoothed_from_real,
                r.basin_real_from_real
            )
            .map_err(|e| format!("write: {}", e))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_steps_small() {
        let s = logarithmic_steps(100);
        assert_eq!(s[0], 1);
        assert_eq!(s.last().copied(), Some(100));
        assert!(s.contains(&9));
        assert!(s.contains(&10));
        assert!(s.contains(&50));
        assert!(s.contains(&90));
    }

    #[test]
    fn test_log_steps_appends_max_when_not_decade() {
        let s = logarithmic_steps(150);
        assert_eq!(s.last().copied(), Some(150));
        assert!(s.contains(&100));
    }

    #[test]
    fn test_execute_runs() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec { kind: GraphKind::Random, n: 30, d: 4.0, seed: 0 };
        let stored = StoredGraph::generate(spec);
        let prob = stored.problem();
        let mut cfg = RunConfig::new("t");
        cfg.log10_iterations = 2;
        cfg.theta = Some(0.0);
        let r = execute(spec, &cfg, &prob, 42);
        assert!(!r.records.is_empty());
        assert_eq!(r.records[0].step, 0);
    }
}
