# gpp_utils

グラフ分割問題 (Graph Partitioning Problem) を焼きなまし法で解くための Rust ライブラリ。

## 概要

グラフの頂点集合を2つの部分集合に分割し、**カットエッジ数**と**バランスペナルティ**の合計を最小化する組合せ最適化問題を扱う。汎用的な最適化トレイトを軸に設計されており、グラフ分割以外の問題にも拡張可能。

## モジュール構成

```
src/
├── lib.rs                  # クレートルート
├── optimization.rs         # 汎用最適化トレイト
├── graph_partition.rs      # グラフ分割問題の定義
├── simulated_annealing.rs  # 焼きなまし法ソルバー
└── file_utils.rs           # JSON 入出力ユーティリティ
```

### optimization — 最適化問題トレイト

`OptimizationProblem<P, S, G>` トレイトは、問題・解・生成方法の3つの型パラメータを取る汎用インターフェースを定義する。

| 型パラメータ | 意味 | グラフ分割での具体型 |
|---|---|---|
| `P` | 問題インスタンス | `Graph` |
| `S` | 解の表現 | `Partition` (`Vec<bool>`) |
| `G` | 生成方式 | `GraphGenerationMethod` |

#### 必須メソッド

| メソッド | 説明 |
|---|---|
| `new(problem: P) -> Self` | 問題インスタンスからソルバーを構築 |
| `generate_problem(method: G, rng) -> P` | 指定方式で問題を生成 |
| `score(&self, solution: &S) -> f64` | 解の目的関数値（小さいほど良い） |
| `neighbour_size(&self) -> usize` | 近傍のサイズ |
| `neighbour(&self, id, solution, score) -> (S, f64)` | 差分計算による近傍解の生成 |
| `create_random_solution(&self, rng) -> S` | ランダムな初期解を生成 |

#### デフォルト実装メソッド

| メソッド | 説明 |
|---|---|
| `random_neighbour(solution, score, rng) -> (S, f64)` | ランダムに近傍を1つ選択 |
| `basin(solution) -> S` | 最急降下法で局所最適解へ移動 |
| `basin_with_distance(solution) -> (S, usize)` | 局所最適解と移動距離を返す |

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

### simulated_annealing — 焼きなまし法

2種類のソルバーを提供する。

#### SimulatedAnnealingSolver（標準 SA）

定数温度での焼きなまし法。設定項目:

| パラメータ | 型 | 説明 |
|---|---|---|
| `cooling_schedule` | `CoolingSchedule` | 冷却スケジュール（現在は `Constant` のみ） |
| `max_iterations` | `usize` | 最大反復回数 |
| `log_interval` | `Option<usize>` | ログ出力間隔 |

#### AdaptiveSimulatedAnnealingSolver（適応的 SA）

複数プロセスの並列探索と準平衡判定による適応的温度制御を行う。

| パラメータ | 型 | 説明 |
|---|---|---|
| `initial_acceptance_probability` | `f64` | 初期受理確率 p_s（初期温度の自動決定に使用） |
| `epsilon` | `f64` | 準平衡判定の閾値 |
| `gamma` | `f64` | 冷却比（温度に毎ステップ掛ける係数） |
| `num_processes` | `usize` | 並列プロセス数 |
| `max_steps` | `usize` | 最大総探索ステップ数 |

主な特徴:

- **初期温度の自動決定**: 目標受理確率 p_s に最も近い温度を、100回の試行SA実行から推定する
- **準平衡判定**: プロセス間のコスト平均の分散（Ω）が初期値の ε 倍未満に収束したら温度を下げる
- **比熱の計算**: 各温度段階でのコスト分散から比熱 C(T) = Var(f) / T^2 を計算
- **実験記録**: 指定ステップでの解・比熱データを JSON ファイルに出力可能

### file_utils — ファイルユーティリティ

| 関数 | 説明 |
|---|---|
| `save_json(data, path)` | Serialize 可能なデータを整形 JSON で保存 |
| `load_json(path)` | JSON ファイルを読み込みデシリアライズ |
| `ensure_dir_exists(path)` | ディレクトリが存在しなければ作成 |

## 使い方

### 依存関係の追加

```toml
[dependencies]
gpp_utils = { path = "../gpp_utils" }
```

### 基本的な使用例

```rust
use gpp_utils::graph_partition::{Graph, GraphPartitionProblem, GraphGenerationMethod};
use gpp_utils::optimization::OptimizationProblem;
use gpp_utils::simulated_annealing::{SimulatedAnnealingSolver, SimulatedAnnealingConfig};
use rand_mt::Mt19937GenRand64;

fn main() {
    let mut rng = Mt19937GenRand64::new(42);

    // ランダムグラフの生成
    let method = GraphGenerationMethod::Random {
        node_count: 100,
        expected_degree: 5.0,
    };
    let graph = GraphPartitionProblem::generate_problem(method, &mut rng);
    let problem = GraphPartitionProblem::new(graph);

    // 初期解の生成
    let initial_solution = problem.create_random_solution(&mut rng);

    // 焼きなまし法で解く
    let config = SimulatedAnnealingConfig::new_constant(10.0, 100_000);
    let solver = SimulatedAnnealingSolver::new(config);
    let (best_solution, stats) = solver.solve(&problem, initial_solution, &mut rng);

    println!("初期スコア: {}", stats.initial_score);
    println!("最良スコア: {}", stats.best_score);
}
```

### 適応的 SA の使用例

```rust
use gpp_utils::simulated_annealing::{AdaptiveSimulatedAnnealingSolver, AdaptiveSAConfig};

let config = AdaptiveSAConfig::new(
    0.5,   // 初期受理確率
    0.3,   // 準平衡閾値 epsilon
    0.75,  // 冷却比 gamma
    8,     // 並列プロセス数
    1_000_000, // 最大ステップ数
);
let solver = AdaptiveSimulatedAnnealingSolver::new(config);
let (best_solution, stats) = solver.solve(&problem, initial_solution, &mut rng);
```

## 依存クレート

| クレート | バージョン | 用途 |
|---|---|---|
| `rand` | 0.8 | 乱数生成 |
| `rand_mt` | 4.2 | Mersenne Twister (MT19937) による再現可能な乱数 |
| `serde` | 1.0 | シリアライズ / デシリアライズ |
| `serde_json` | 1.0 | JSON 入出力 |
