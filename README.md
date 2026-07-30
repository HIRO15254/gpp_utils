# gpp_utils

グラフ分割問題 (Graph Partitioning Problem) を焼きなまし法で解くための Rust ライブラリ。

## 概要

グラフの頂点集合を2つの部分集合に分割し、**カットエッジ数**と**バランスペナルティ**の合計を最小化する組合せ最適化問題を扱う。汎用的な最適化トレイトを軸に設計されており、グラフ分割以外の問題にも拡張可能。

> **アルゴリズム解説**: 目的関数の定式化、差分評価の導出、各ソルバー
> （山登り / SA / EO / SQA）とスムージング戦略の数学的背景および実装の
> 詳細は **[docs/algorithms.md](docs/algorithms.md)** にまとめている。

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
├── batch.rs                # GUI/CLI 共通のバッチ実行ランナー
├── bin/gui.rs              # 4 タブ構成の実験 GUI
└── bin/cli.rs              # ヘッドレスなバッチ実行 CLI
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
| `ExtremalOptimizationSolver` | τ-EO（汎用版）。べき乗則確率で低適応度の構成要素を変更 |
| `SimulatedQuantumAnnealingSolver` | 鈴木–トロッター分解による SQA（P レプリカ） |

> 実験ワークフロー（GUI / CLI / batch）は汎用トレイトではなく `run_executor::execute` を使う。近傍（フリップ / スワップ）× 受理規則（メトロポリス / EO ランク）の 2×2:
> - フリップ近傍: `Sa`（`run_sa_*`、smoothing 対応・実ベイスン）↔ `EoFlip` 系（`run_eo_flip`。適応度は `EoFlipFitnessSpec` で Legacy / MulAlpha / AddBeta / MulGamma の 4 方式）
> - スワップ近傍・厳密バランス: `SaSwap`（`run_sa_swap`）↔ `Eo`（`run_eo`、スペック忠実）。ベイスンは `hill_climb_swap_fast`（スワップ最急降下）で算出
>
> `ExtremalOptimizationSolver` はバランス・スワップ概念を持たない汎用トレイト版で、ワークフローでは使われない。

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

### run_config — 実行条件

```rust
pub struct RunConfig {
    pub name: String,
    pub theta: Option<f64>,      // 温度を Theta = log10(T) で指定。None なら T = 0（SA のみ）
    pub log10_iterations: u32,   // 反復回数 = 10^N
    pub smoothing: SmoothingSpec,
    pub solver: SolverSpec,      // 既定 = Sa。Eo { tau } のとき theta/smoothing は無視
}
```

- `temperature()` — `theta = None` のとき `0.0`、それ以外は `10^theta`
- `iterations()` — `10^log10_iterations`（最大 `10^9`）
- `id()` — キャッシュキー。SA は従来形式 (`th+0_iter4_kavg8`, `T0_iter5_none`)、
  EO は `eo_iter5_tau1p4` のように独立した名前空間
- `SmoothingSpec` バリアント: `None` / `KAverage(k)` / `RandomKAverage(k)` / `WeightedAverage(w)`
- `SolverSpec` バリアント（7 種）:
  - `Sa` — フリップ近傍・固定温度メトロポリス
  - `SaSwap` — スワップ近傍・厳密バランス・メトロポリス
  - `Eo { tau }` — スワップ近傍・厳密バランスの τ-EO
  - `EoFlip { tau, alpha_eo, diff_exp }` — フリップ近傍版 τ-EO（SA と同一の近傍・目的関数・ベイスン算出を共有）。
    `alpha_eo`（既定 0.05）/ `diff_exp`（p、既定 2.0）は適応度 `q = alpha_eo·(|diff|^p − |diff_after|^p)` の
    係数と指数で、**手選択のみに影響し目的関数は不変**。既定値なら id は従来どおり
    `eoflip_iter5_tau1p4`、非既定のときだけ `_a{α}` / `_p{p}` が付く
  - `EoFlipMulAlpha { tau, alpha }` / `EoFlipAddBeta { tau, beta }` / `EoFlipMulGamma { tau }` —
    フリップ近傍版 τ-EO の適応度バリエーション（`λ0 = g/deg` × 多数派/少数派インジケータ λ1。
    [アルゴリズム 5.3.2 節](docs/algorithms.md)）。id は `eoflipmulalpha_iter{N}_tau{τ}_a{α}` /
    `eoflipaddbeta_iter{N}_tau{τ}_b{β}` / `eoflipmulgamma_iter{N}_tau{τ}`

  τ 既定 1.4、温度は `theta`。`#[serde(default)]` なので `solver` を持たない既存 JSON は `Sa` として読まれる。
  → フリップ近傍では `Sa`↔`EoFlip` 系、スワップ近傍では `SaSwap`↔`Eo` の 2×2 比較。
  全 EO 共通で、同率適応度は「同率群の合計重み = 非同率時の合計、群内等分」の平均化規則で扱う
  （`run_executor::select_eo_rank`、[アルゴリズム 5.3.3 節](docs/algorithms.md)）

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
2. **Configs** — `RunConfig` のリストを編集する。各設定で `Solver`（`SA (flip)` / `SA swap` / `EO swap` / `EO flip` / `EO flip mul-alpha` / `EO flip add-beta` / `EO flip mul-gamma`）を選ぶ。`SA (flip)` / `SA swap` では `use Theta` チェックボックスで `Theta` を有効化（無効なら `T = 0` の貪欲）。`SA (flip)` のみスムージング種別と K も選択できる（`SA swap` は smoothing なし）。EO 系を選ぶと `tau` スライダ（1.0〜2.0、既定 1.4）が出て、theta/smoothing 行は無視される。`EO flip mul-alpha` / `EO flip add-beta` ではさらに `alpha` / `beta` の数値入力が出る（`EO flip` の `alpha_eo` / `diff_exp` は GUI では既定値固定、バッチ JSON でのみ指定可）。`log10(iter)` スライダは全ソルバ共通。`Generate from sweep` を開くと、Solver を選んで SA は温度（＋flipはsmoothing/K）を、EO 系は τ をカンマ区切りで複数指定し、その総当たり組み合わせを設定リストへ一括追加できる（α_eo/p/α/β の sweep 軸はバッチ JSON 専用）。
3. **Run** — 選択中のグラフ・チェック済み Config・`start_seed` / `# seeds` で一括実行する。実行は裏スレッドで進み、`ResultStore` にキャッシュされた `(graph, config, seed)` 三つ組はスキップされる。プログレスバーとログで進捗を確認できる。`Export batch JSON` ボタンで、同じ選択内容を CLI 用バッチ定義 (`data/batch.json`) として書き出せる（`cli --batch data/batch.json` でそのまま再実行可能）。
4. **Results** — 現在の選択（グラフ・Config・seed 範囲）にマッチする結果を `Load matching` で読み込み、6 トレースを log-step 軸でプロット。各トレースはチェックボックスで個別に表示切替できる。`Export TSV` で選択結果を `data/tsv/<graph_id>/<config_id>/seed_<seed>.tsv` に書き出す。

ディレクトリ構成:

```
data/
├── graphs/<graph_id>.json            # 生成済みグラフ
├── results/<graph_id>/<config_id>/   # 実行結果 JSON
└── tsv/<graph_id>/<config_id>/       # gnuplot 用 TSV
```

### CLI（`cargo run --bin cli`）

GPU やディスプレイの無い環境（Azure VM など）向けのヘッドレスなバッチ実行。
GUI の Run タブと**同じ実行パス**（`batch::run_batch` → `run_executor::execute`）を
共有しているため、最適化の速度・出力フォーマットは GUI と完全に一致する。結果は
`data/results/` に GUI と同一レイアウトで保存され、ローカルの GUI Results タブで
そのまま閲覧できる。

```
cargo run --release --bin cli -- --batch <定義>.json [オプション]
```

| オプション | 既定値 | 説明 |
|---|---|---|
| `--batch <FILE>` | （必須） | JSON バッチ定義ファイル |
| `--out <DIR>` | `data/results` | 結果 JSON の保存先 |
| `--graphs <DIR>` | `data/graphs` | グラフのロード／生成キャッシュ先 |
| `--threads <N>` | 論理コア数 | 並列ワーカ数 |
| `--overwrite` | （オフ） | 既存結果も上書き再計算する（既定は既存をスキップ） |

#### バッチ定義 JSON の書式

`graphs × configs × seeds` の直積がジョブとして実行される。実行する設定は
`configs`（明示列挙）と `config_sweep`（総当たり展開）の連結で、どちらか一方だけでも
両方でもよい。サンプルは [`examples/batch.example.json`](examples/batch.example.json)
（明示列挙）、[`examples/batch.sweep.example.json`](examples/batch.sweep.example.json)
（sweep）、[`examples/batch.eo.example.json`](examples/batch.eo.example.json)
（τ-EO）を参照。

```json
{
  "graphs": [
    { "kind": "Random", "n": 124, "d": 5.0, "seed": 0 }
  ],
  "configs": [
    { "name": "T=1, 10^4, none", "theta": 0.0, "log10_iterations": 4, "smoothing": "None" },
    { "name": "greedy, kavg8", "theta": null, "log10_iterations": 4, "smoothing": { "KAverage": 8 } }
  ],
  "seed_start": 0,
  "seed_count": 3
}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `graphs` | 配列 | 実行対象グラフの仕様。未生成なら自動生成・キャッシュされる |
| `graphs[].kind` | `"Random"` / `"Geometric"` | グラフ生成方式 |
| `graphs[].n` | 整数 | 頂点数 |
| `graphs[].d` | 実数 | 期待次数 |
| `graphs[].seed` | 整数 | グラフ生成シード |
| `configs` | 配列（任意） | 明示列挙する実行条件 `RunConfig`。省略可 |
| `configs[].name` | 文字列 | 表示用ラベル（キャッシュキー `id()` には影響しない） |
| `configs[].theta` | 実数 / `null` | 温度 Θ = log10(T)。`null` で T = 0（貪欲）。EO では無視 |
| `configs[].log10_iterations` | 整数 | 反復回数 = 10^N |
| `configs[].smoothing` | 下表参照 | スムージング戦略。EO では無視 |
| `configs[].solver` | `"Sa"` / `"SaSwap"`（スワップ近傍SA）/ `{ "Eo": { "tau": 1.4 } }`（スワップ版EO）/ `{ "EoFlip": { "tau": 1.4, "alpha_eo": 0.05, "diff_exp": 2.0 } }`（フリップ版EO、`alpha_eo`/`diff_exp` は省略時既定値）/ `{ "EoFlipMulAlpha": { "tau": 1.4, "alpha": 0.5 } }` / `{ "EoFlipAddBeta": { "tau": 1.4, "beta": 1.0 } }` / `{ "EoFlipMulGamma": { "tau": 1.4 } }`（任意、既定 `"Sa"`） | ソルバー選択 |
| `config_sweep` | オブジェクト（任意） | 総当たり展開する指定（下記） |
| `seed_start` | 整数 | 実行シードの開始値 |
| `seed_count` | 整数 | シード本数（`seed_start, seed_start+1, …` を `seed_count` 個） |

`smoothing`（`SmoothingSpec`）の JSON 表記:

| 戦略 | 表記 |
|---|---|
| 平滑化なし | `"None"` |
| 決定論的 K 近傍平均 | `{ "KAverage": 8 }`（K = 近傍の個数） |
| 確率的 K 近傍平均 | `{ "RandomKAverage": 8 }`（K = 近傍の個数） |
| 重み付き平均 | `{ "WeightedAverage": 0.5 }`（重み w = 0〜1） |

`WeightedAverage` の値は重み `w` で、`w × 全近傍平均 + (1 − w) × 実スコア` のブレンド比。
`w = 0` は平滑化なし相当、`w = 1` は全近傍平均。グラフサイズに依存しない（範囲外は 0〜1 にクランプ）。

`solver` を `Eo`（厳密バランスのスワップ版）または `EoFlip` 系（フリップ近傍版＝SA と同一の
近傍・目的関数・ベイスンを共有）にすると τ-EO で実行する（[アルゴリズム 5.3 節](docs/algorithms.md)）。
いずれも `theta` / `smoothing` は無視される。

```json
{ "name": "eo-swap tau=1.4",  "theta": null, "log10_iterations": 5, "smoothing": "None", "solver": { "Eo":     { "tau": 1.4 } } }
{ "name": "eo-flip tau=1.4",  "theta": null, "log10_iterations": 5, "smoothing": "None", "solver": { "EoFlip": { "tau": 1.4 } } }
{ "name": "eo-flip tuned",    "theta": null, "log10_iterations": 5, "smoothing": "None", "solver": { "EoFlip": { "tau": 1.4, "alpha_eo": 0.1, "diff_exp": 0.5 } } }
{ "name": "eo-flip mul-alpha","theta": null, "log10_iterations": 5, "smoothing": "None", "solver": { "EoFlipMulAlpha": { "tau": 1.4, "alpha": 0.5 } } }
{ "name": "eo-flip add-beta", "theta": null, "log10_iterations": 5, "smoothing": "None", "solver": { "EoFlipAddBeta":  { "tau": 1.4, "beta": 1.0 } } }
{ "name": "eo-flip mul-gamma","theta": null, "log10_iterations": 5, "smoothing": "None", "solver": { "EoFlipMulGamma": { "tau": 1.4 } } }
```

#### `config_sweep` — パラメータ総当たり

SA では `thetas × log10_iterations × (smoothing_kind × ks)` の直積、EO 系では
`log10_iterations × taus` に種別固有の軸（`EoFlip`: `× alpha_eos × diff_exps`、
`EoFlipMulAlpha`: `× mul_alphas`、`EoFlipAddBeta`: `× add_betas`）を掛けた直積を取り、
自動的に `RunConfig` 群を生成する。多数のパラメータを一括で網羅実験したいときに使う。

```json
{
  "graphs": [
    { "kind": "Random", "n": 124, "d": 5.0, "seed": 0 }
  ],
  "config_sweep": {
    "thetas": [-1.0, 0.0, 1.0, null],
    "log10_iterations": [4, 5],
    "smoothing_kind": "KAverage",
    "ks": [4, 8, 16]
  },
  "seed_start": 0,
  "seed_count": 3
}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `thetas` | 配列 | 温度 Θ の候補。`null` は T = 0（貪欲）。`solver_kind = "Eo"` では無視 |
| `log10_iterations` | 整数配列 | 反復回数指数 N（= 10^N）の候補 |
| `smoothing_kind` | 下表参照 | 平滑化の種別（生成される全設定で共通）。`solver_kind = "Eo"` では無視 |
| `ks` | 整数配列 | K 近傍個数の候補。`smoothing_kind` が `"KAverage"` / `"RandomKAverage"` のとき使用 |
| `weights` | 実数配列 | 重み（0〜1）の候補。`smoothing_kind` が `"WeightedAverage"` のとき使用 |
| `solver_kind` | `"Sa"` / `"SaSwap"` / `"Eo"` / `"EoFlip"` / `"EoFlipMulAlpha"` / `"EoFlipAddBeta"` / `"EoFlipMulGamma"`（任意、既定 `"Sa"`）。`SaSwap` は `thetas × log10_iterations`、EO 系は `log10_iterations × taus`（× 種別固有の軸）を展開 | ソルバー種別 |
| `taus` | 実数配列（任意） | τ の候補。EO 系のとき直積軸に使う（空なら既定 1.4 の 1 通り） |
| `alpha_eos` | 実数配列（任意） | `EoFlip` の適応度係数 α_eo の候補（空なら既定 0.05 の 1 通り。`EoFlip` 以外では無視） |
| `diff_exps` | 実数配列（任意） | `EoFlip` の適応度指数 p（diff_exp）の候補（空なら既定 2.0 の 1 通り。`EoFlip` 以外では無視） |
| `mul_alphas` | 実数配列（任意） | `EoFlipMulAlpha` の係数 α の候補（空なら既定 1.0 の 1 通り。それ以外では無視） |
| `add_betas` | 実数配列（任意） | `EoFlipAddBeta` の係数 β の候補（空なら既定 1.0 の 1 通り。それ以外では無視） |

`smoothing_kind`（`SmoothingKind`）の表記: `"None"` / `"KAverage"` / `"RandomKAverage"` / `"WeightedAverage"`。
`KAverage` / `RandomKAverage` は `ks` を、`WeightedAverage` は `weights` を直積軸に使う（`None` はどちらも無視）。

SA の例は `4 thetas × 2 iterations × 3 Ks = 24` 設定に展開され、`1 graph × 24 configs × 3 seeds = 72` ジョブが実行される。EO の sweep は次のように書く（`2 iterations × 3 taus = 6` 設定）:

```json
"config_sweep": {
  "log10_iterations": [4, 5],
  "solver_kind": "Eo",
  "taus": [1.2, 1.4, 1.6]
}
```

GUI では Configs タブの「Generate from sweep」で Solver を選ぶと同じ展開を行える。

既に結果が存在する `(graph, config, seed)` 三つ組は既定でスキップされる
（`--overwrite` で再計算）。グラフのロード／生成や保存に失敗した場合は
標準エラーに出力し、終了コードを非 0 にする。

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
| `clap` | 4 | コマンドライン引数の解析（`bin/cli.rs`） |
