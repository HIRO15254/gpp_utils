# gpp_utils

グラフ分割実験を行う Rust ライブラリと `gpp` CLI です。無向・重みなしグラフを2群へ分け、`cut_edges + alpha * (size_a - size_b)^2` を最小化します。

## 対応範囲

- グラフ生成: Random (Erdos-Renyi) / Geometric
- 近傍: Flip（任意分割）/ Swap（偶数頂点・等分割）
- 探索: Hill Climbing、固定温度 Simulated Annealing、Extremal Optimization
- 平滑化: `none`、`all_average`、`random_k_average`、`weighted_average`（HC/SAのみ）
- EO: 必須の `taus` と適応度 `default`（`good_edge_fraction`）。Rustの登録機構で拡張可能
- TOML/JSON設定、直積スイープ、CPU並列、キャンセル、ジョブ単位の再開
- 結果はJSON正本、必要時にTSVを生成。集計・作図は外部ツールで行う

GUI、SQA、連続緩和、`k_average`は提供しません。詳細は [docs/application-plan.md](docs/application-plan.md)、保存形式は [docs/output-format.md](docs/output-format.md)、数式は [docs/algorithms.md](docs/algorithms.md)を参照してください。

計算結果を維持する高速化と再測定方法は [docs/performance.md](docs/performance.md)に記載しています。

## ビルド

```text
cargo build --release --bin gpp
```

ビルドのみの場合は`gpp`を`cargo run --release --bin gpp --`に置き換えて実行できます。PATHへインストールする場合は`cargo install --path . --bin gpp`を使います。

## CLIの最短手順

```text
gpp init experiment.toml
gpp validate experiment.toml
gpp plan experiment.toml
gpp run experiment.toml --root data/v1 --threads 4
gpp inspect --batch <表示されたbatch_id> --root data/v1
gpp resume --batch <batch_id> --root data/v1 --threads 4
gpp export --batch <batch_id> --root data/v1 --out export
```

`gpp plan experiment.toml --out experiment.json`は既定値とバージョンを固定した未展開条件を保存します。保存した条件は`gpp run --experiment experiment.json --root data/v1 --threads 4`で実行します。

## 設定例

```toml
schema_version = 1
run_seeds = [0]
neighborhoods = ["flip"]

[problem]
alpha = 0.05

[budget]
max_steps = 1000

[[graphs]]
kind = "random"
node_counts = [32]
expected_degrees = [4.0]
seeds = [42]

[[solvers]]
kind = "sa"
temperatures = [1.0]
[[solvers.smoothing]]
kind = "none"
```

比較実験では配列を増やして全組み合わせを実行します。例えば近傍、SAの温度、平滑化、EOの`taus`、`run_seeds`を列挙します。Swapを使うグラフの頂点数は偶数にしてください。EOの`taus`は必須です。

## 保存される結果

```text
data/v1/
├── graphs/<graph_id>.json
├── batches/<batch_id>/experiment.json
└── runs/<condition_id>/
    ├── seed_<seed>.json
    └── seed_<seed>.incomplete.json
```

各実行JSONは、重複排除した`partitions`配列と、終了時の現行解`final_solution`、終了時の暫定解`best_solution`、各計測点の`current_solution` / `best_solution`参照を保存します。暫定解はそのステップまでに実評価値が最小だった現行解です。同点では先に得た解を維持します。実評価値、カット数、群サイズ、ペナルティなど分割から求められる値は保存せず、読み込み時または`export`時に算出します。

`export`は`runs.tsv`、`traces.tsv`、`metadata.json`を生成します。これらは再生成可能な分析用ファイルで、通常の実験実行では作成されません。

## ライブラリ利用

実験APIは設定を検証・展開してから単一実行またはバッチ実行へ渡します。代表的な入口は次のとおりです。

```rust
use gpp_utils::{compile_experiment, run_one};
use gpp_utils::experiment::plan::load_spec;
use gpp_utils::fitness::FitnessRegistry;
use gpp_utils::graph_partition::Graph;
use gpp_utils::optimization::CancellationToken;

fn main() -> anyhow::Result<()> {
    let plan = compile_experiment(load_spec(std::path::Path::new("experiment.toml"))?)?;
    let job = &plan.jobs[0];
    let cancel = CancellationToken::new();
    let graph = Graph::generate(&job.condition.graph, &cancel)?;
    let result = run_one(&graph, &job.condition, job.seed, &cancel, &FitnessRegistry::default())?;
    println!("steps={}", result.completed_steps);
    Ok(())
}
```

実行可能な例は`cargo run --release --example basic_usage`、探索速度の測定は`cargo run --release --example bench_search`で実行できます。`RunView`が保存された解参照から派生値を生成します。

## 開発

```text
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
cargo build --release --all-targets
```

実装方針、サブエージェントの並列分担、GPT-5.6 Luna/Terra/Solの複雑さ別利用方針は [AGENTS.md](AGENTS.md)に記載しています。
