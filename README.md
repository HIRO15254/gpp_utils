# gpp_utils

グラフ分割問題 (Graph Partitioning Problem) を焼きなまし法で解くための Rust ライブラリ。

## 概要

グラフの頂点集合を2つの部分集合に分割し、**カットエッジ数**と**バランスペナルティ**の合計を最小化する組合せ最適化問題を扱う。汎用的な最適化トレイトを軸に設計されており、グラフ分割以外の問題にも拡張可能。

## モジュール構成

```
src/
├── lib.rs                  # クレートルート
├── optimization.rs         # 汎用最適化トレイト (Problem / Smoothing / Solver)
├── graph_partition.rs      # グラフ分割問題の定義
├── smoothing.rs            # スムージング戦略 (None / KAvg / RandomKAvg / Weighted)
├── solvers/                # ソルバー群 (HC / SA / EO / SQA)
├── experiment.rs           # ベイスン評価などの補助
├── file_utils.rs           # JSON 入出力ユーティリティ
├── graph_spec.rs           # グラフ仕様とライブラリ (data/graphs に永続化)
├── run_config.rs           # SA 実行条件 (Theta / 10^N 反復 / Smoothing)
├── run_executor.rs         # 対数刻み 6 トレース計測と結果ストア
└── bin/gui.rs              # 4 タブ構成の実験 GUI
```

### optimization — 最適化フレームワークのトレイト

問題・スコア計算・探索戦略を独立した3つのトレイトに分離し、任意の組み合わせで
最適化を実行できる構造。

| トレイト | 役割 |
|---|---|
| `Problem<S>` | 解 `S` のスコア計算・近傍生成・初期解生成を定義する問題インスタンス |
| `Smoothing<S>` | スコア評価方法の差し替え層（実スコア / K-近傍平均など） |
| `Solver` | `Problem` と `Smoothing` を受け取り最適化を実行する探索戦略 |

`Problem<S>` の主なメソッド:

| メソッド | 説明 |
|---|---|
| `score(&self, solution) -> f64` | 解の目的関数値（小さいほど良い） |
| `neighbour(&self, solution) -> Vec<S>` | 全近傍を生成 |
| `neighbour_size(&self) -> usize` | 近傍サイズ |
| `random_solution(&self, rng) -> S` | ランダムな初期解を生成 |
| `score_at_move(&self, solution, idx) -> f64` | 移動 `idx` 適用後のスコア（差分計算でオーバーライド可能） |
| `apply_move(&self, solution, idx)` | 移動 `idx` を in-place 適用 |

ソルバーは実行統計 `SolverStats`（反復数・初期/最終/最良スコア・受理/棄却数・
スコア履歴）を返す。

### graph_partition — グラフ分割問題

#### 目的関数

```
score = カットエッジ数 + ALPHA * (|V1| - |V2|)^2
```

- **カットエッジ**: 異なるパーティションに属する端点を持つ辺の数
- **バランスペナルティ**: パーティションサイズ差の二乗に係数 `ALPHA`（デフォルト 0.05）を掛けた値

#### 近傍構造

1つの頂点のパーティション割り当てを反転（フリップ）する操作。`neighbour()` ではスコアの差分計算（Δ評価）により O(deg(v)) で効率的に新スコアを求める。

#### グラフ生成方式

| バリアント | モデル | 説明 |
|---|---|---|
| `Random` | Erdos-Renyi G(n, p) | 各辺を独立に確率 p = expected_degree / (n-1) で生成 |
| `Geometric` | 幾何ランダムグラフ | [0,1]^2 上に頂点を配置し、距離が閾値以下の頂点対に辺を張る |

#### 主な型

- `Graph` — 無向・重みなしグラフの隣接リスト表現
- `Partition` — `Vec<bool>` の型エイリアス
- `GraphPartitionProblem` — `OptimizationProblem` の実装

### solvers — ソルバー群

`Solver` トレイトを実装する4種類の探索戦略。いずれも `Problem` と `Smoothing` の
任意の組み合わせで動作する。

| ソルバー | 説明 |
|---|---|
| `HillClimbingSolver` | 貪欲な局所探索。改善がなくなるまで最良近傍へ移動 |
| `SimulatedAnnealingSolver` | 固定温度のメトロポリス法による焼きなまし |
| `ExtremalOptimizationSolver` | τ-EO。べき乗則確率で低適応度の構成要素を変更 |
| `SimulatedQuantumAnnealingSolver` | 鈴木–トロッター分解による SQA（P レプリカ） |

### smoothing — スムージング戦略

`Smoothing` トレイトの実装。スコアランドスケープを平滑化する。

| 戦略 | 説明 |
|---|---|
| `NoSmoothing` | 平滑化なし（実スコアをそのまま使用） |
| `KAveragingSmoothing` | 先頭 K 近傍のスコア平均（決定論的） |
| `AllNeighbourAveragingSmoothing` | 全近傍のスコア平均 |
| `RandomKSmoothing` | ランダム K 近傍平均（距離2近傍フォールバックあり） |
| `WeightedNeighbourSmoothing` | K/n × 全近傍平均 + (1−K/n) × 実スコア の線形ブレンド |

### file_utils — ファイルユーティリティ

| 関数 | 説明 |
|---|---|
| `save_json(data, path)` | Serialize 可能なデータを整形 JSON で保存 |
| `load_json(path)` | JSON ファイルを読み込みデシリアライズ |
| `ensure_dir_exists(path)` | ディレクトリが存在しなければ作成 |

## 実験ワークフロー

`graph_spec` / `run_config` / `run_executor` の 3 モジュールは、**プリセット
された条件で大量の SA 実行を回し、対数刻みでスナップショットを取り、
ファイルにキャッシュする** ことを目的とした実験用バックボーン。GUI（`bin/gui.rs`）
の 4 タブ（Graphs / Configs / Run / Results）はこのバックボーンの上で動く。

### graph_spec — グラフ仕様とライブラリ

プリセット定数:

| 定数 | 値 |
|---|---|
| `NODE_COUNTS` | `[62, 124, 250, 500, 1000, 2000]` |
| `EXPECTED_DEGREES` | `[2.5, 5.0, 10.0, 20.0, 40.0]` |
| `GraphKind` | `Random` / `Geometric` |

主な型・メソッド:

- `GraphSpec { kind, n, d, seed }` — グラフの一意キー。`id()` 例: `random_n124_d5_s7`, `geom_n62_d2p5_s0`
- `StoredGraph` — 隣接リストと（`Geometric` の場合）座標を保持し JSON で永続化
- `GraphLibrary::load_or_generate(spec)` — 既存ファイルがあれば読み込み、無ければ生成して保存（保存先は `data/graphs/<id>.json`）
- `GraphLibrary::list()` — ディレクトリ内のグラフを列挙

### run_config — SA 実行条件

```rust
pub struct RunConfig {
    pub name: String,
    pub theta: Option<f64>,      // 温度を Theta = log10(T) で指定。None なら T = 0
    pub log10_iterations: u32,   // 反復回数 = 10^N
    pub smoothing: SmoothingSpec,
}
```

- `temperature()` — `theta = None` のとき `0.0`、それ以外は `10^theta`
- `iterations()` — `10^log10_iterations`（最大 `10^9`）
- `id()` — キャッシュキー。例: `th+0_iter4_kavg8`, `T0_iter5_none`
- `SmoothingSpec` バリアント: `None` / `KAverage(k)` / `RandomKAverage(k)` / `WeightedAverage(k)`

### run_executor — 対数刻み実行と結果ストア

各スナップショットで 1 つの `StepRecord` を記録する:

| フィールド | 内容 |
|---|---|
| `step` | SA のステップ番号（0 = 初期解、その後 1, 2, …, 9, 10, 20, …） |
| `current_smoothed` | 現在解のスムージング空間でのスコア |
| `current_real` | 現在解の元空間（実）スコア |
| `basin_smoothed_from_smoothed` | スムージング空間で山登り後のベイスンのスムージング空間スコア |
| `basin_real_from_smoothed` | 同ベイスンの元空間スコア |
| `basin_smoothed_from_real` | 元空間で山登り後のベイスンのスムージング空間スコア |
| `basin_real_from_real` | 同ベイスンの元空間スコア |

主な API:

- `logarithmic_steps(max_iter)` — `1, 2, ..., 9, 10, 20, ...` のサンプリング点列を返す（必要に応じて末尾に `max_iter` を追加）
- `execute(spec, cfg, prob, seed) -> RunResult` — 単一シードを実行し、初期＋対数刻みでスナップショットを記録
- `ResultStore::path_for(spec, cfg, seed)` → `data/results/<graph_id>/<config_id>/seed_<seed>.json`
- `ResultStore::exists / load / save` — 結果のキャッシュ管理（GUI は完了済みの triple をスキップする）
- `ResultStore::export_tsv(result, path)` — gnuplot 互換の TSV を出力（列: `step, cur_sm, cur_real, basin_sm_from_sm, basin_real_from_sm, basin_sm_from_real, basin_real_from_real`）

### GUI（`cargo run --bin gui`）

4 つのタブで実験を回す:

1. **Graphs** — `kind` / `N` / `D` / `seed` を選んで `Generate / Load`。既に `data/graphs` に同 ID のグラフがあれば再利用、無ければ生成して保存。下部のリストから 1 つ選ぶと可視化される。
2. **Configs** — `RunConfig` のリストを編集する。`use Theta` チェックボックスで `Theta` を有効化（無効なら `T = 0` の貪欲）、`log10(iter)` スライダで反復数を設定、スムージング種別と K を選択。
3. **Run** — 選択中のグラフ・チェック済み Config・`start_seed` / `# seeds` で一括実行する。実行は裏スレッドで進み、`ResultStore` にキャッシュされた `(graph, config, seed)` 三つ組はスキップされる。プログレスバーとログで進捗を確認できる。
4. **Results** — 現在の選択（グラフ・Config・seed 範囲）にマッチする結果を `Load matching` で読み込み、6 トレースを log-step 軸でプロット。各トレースはチェックボックスで個別に表示切替できる。`Export TSV` で選択結果を `data/tsv/<graph_id>/<config_id>/seed_<seed>.tsv` に書き出す。

ディレクトリ構成:

```
data/
├── graphs/<graph_id>.json            # 生成済みグラフ
├── results/<graph_id>/<config_id>/   # 実行結果 JSON
└── tsv/<graph_id>/<config_id>/       # gnuplot 用 TSV
```

## 使い方

### 依存関係の追加

```toml
[dependencies]
gpp_utils = { path = "../gpp_utils" }
```

### 基本的な使用例（ソルバー API）

```rust
use gpp_utils::graph_partition::{GraphGenerationMethod, GraphPartitionProblem};
use gpp_utils::optimization::{Problem, Solver};
use gpp_utils::smoothing::NoSmoothing;
use gpp_utils::solvers::HillClimbingSolver;
use rand_mt::Mt19937GenRand64;

fn main() {
    let mut rng = Mt19937GenRand64::new(42);

    // ランダムグラフから問題インスタンスを生成
    let method = GraphGenerationMethod::Random {
        node_count: 100,
        expected_degree: 5.0,
    };
    let problem = GraphPartitionProblem::generate(method, &mut rng);

    // 初期解を生成
    let initial = problem.random_solution(&mut rng);

    // 山登り法で解く（スムージングなし、seed = 42）
    let solver = HillClimbingSolver::new();
    let (_best, stats) = solver.solve(&problem, &NoSmoothing, initial, 42);

    println!("初期スコア: {}", stats.initial_score);
    println!("最良スコア: {}", stats.best_score);
}
```

### 実験ワークフローの使用例（run_executor）

```rust
use gpp_utils::graph_spec::{GraphKind, GraphSpec, StoredGraph};
use gpp_utils::run_config::{RunConfig, SmoothingSpec};
use gpp_utils::run_executor::execute;

// グラフ仕様を決め、生成（既存なら data/graphs からロード）
let spec = GraphSpec { kind: GraphKind::Random, n: 500, d: 5.0, seed: 0 };
let stored = StoredGraph::generate(spec);
let problem = stored.problem();

// SA 実行条件: Θ = 0 (T = 1)、10^4 反復、K = 8 の決定論的スムージング
let mut cfg = RunConfig::new("example");
cfg.theta = Some(0.0);
cfg.log10_iterations = 4;
cfg.smoothing = SmoothingSpec::KAverage(8);

// 対数刻みでスナップショットを取りながら実行（seed = 42）
let result = execute(spec, &cfg, &problem, 42);
println!("記録されたスナップショット数: {}", result.records.len());
```

## 依存クレート

| クレート | バージョン | 用途 |
|---|---|---|
| `rand` | 0.8 | 乱数生成 |
| `rand_mt` | 4.2 | Mersenne Twister (MT19937) による再現可能な乱数 |
| `serde` | 1.0 | シリアライズ / デシリアライズ |
| `serde_json` | 1.0 | JSON 入出力 |
| `eframe` | 0.31 | ネイティブ GUI フレームワーク（`bin/gui.rs`） |
| `egui_plot` | 0.31 | プロット描画（`bin/gui.rs`） |
