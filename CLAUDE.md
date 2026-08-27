# gpp_utils — リポジトリ共通コンテキスト

グラフ分割問題（2 分割、目的 = カット辺数 + 0.05·(|T|−|F|)²）を SA / EO で解く Rust ライブラリ + 実験基盤。
アルゴリズムの詳細は `docs/algorithms.md`、コード構成は `README.md`。

## 実験のベースライン・パラメータセット（2026-08-27 確定）

**今後の実験は、特に指示がない限りこのセットをベースにする。**
プログラム上の単一の真実の源は `experiments_sa_eo/grid.py`（`thetas()` / `taus()` /
`graphs()` / `RUN_SEEDS` / `cfg_*()`）。新しい実験は grid.py を import して組み立て、
ディレクトリ名の正規表現ではなく `grid.family_of(cfg)` / `grid.param_of(cfg)` で系列を判定する。

### 問題例（72 グラフ）
| 項目 | 値 |
|---|---|
| 種類 `kind` | Random, Geometric |
| 頂点数 `n` | 124, 250, 500 |
| 平均次数 `d` | 5, 10, 20 |
| 生成シード | 0, 1, 2, 3（各組 4 インスタンス） |
| 保存先 | `data/graphs/<id>.json`（id 例: `geom_n500_d10_s2`） |

18 組（kind×n×d）× 4 インスタンス = 72 グラフ。

### ソルバー系列（4 系列 = 近傍 {flip, swap} × 手法 {SA, EO}）
| 系列名 | `solver` | スイープ対象 |
|---|---|---|
| flipSA | `"Sa"` | Θ = log10(T) |
| swapSA | `"SaSwap"` | Θ |
| swapEO | `{"Eo": {"tau": τ}}` | τ |
| flipEO | `{"EoFlipMulAlpha": {"tau": τ, "alpha": 1.0}}`（元論文の λ=g/deg をフリップ近傍へ） | τ |

共通: `log10_iterations = 7`（10^7 ステップ）、`smoothing = "None"`。

### パラメータ格子
| 系列 | 格子 | 点数 |
|---|---|---|
| SA（両近傍） | Θ = −1.50 … +1.50、0.05 刻み | 61 |
| EO（両近傍） | τ = 0.00 … 1.70、0.05 刻み + 1.85, 2.00 | 37 |

### 反復・評価
- 実行シード 0..31（32 本）→ 1 条件 = 4 インスタンス × 32 シード = **128 試行**。
- 1 ラウンド = 実行シード 1 本ぶんを全条件に追加する方式で回す
  （`make_batches.py` / `run_rounds.sh`）。途中で止めても全条件が同数になる。
- 主指標: 最終ステップの `basin_real_from_best`（best-so-far から山登りした盆地値）の 128 試行平均。
  補助: `best_real`、`basin_diff_from_best`、ステップ別トレース（`records`）。
- バイナリは決定的（同一 config+seed は elapsed_ms 以外 bit 一致）なので、
  新しい条件は既存ストアにそのまま追記してよい。`cli --batch` は既存 seed をスキップ（冪等）。

### 完了済みデータ
- ストア `data/results_sa_eo/`: 上記格子の全条件 × 128 試行 = **451,584 run**、欠損なし
  （本実験 2026-08-23〜25 + 追試1/4/5 で両裾を埋めた）。
- 集計: `experiments_sa_eo/quick_agg.py` → `quick_agg.csv`（最終値）、
  `agg_steps.py` → `agg_steps.csv`（ステップ別）。分析まとめは
  `experiments_sa_eo/REPORT_step_analysis.md`。
- 主要な結論: 最適 Θ / τ は 4 系列 × 18 組すべてで格子内部にある（両端の延長は不要）。
  swapEO ≲ flipSA < flipEO < swapSA（全体順位）。geom 疎グラフでは EO が SA を大差で上回る。

### 補助ストア（本ストアに混ぜない）
- `data/results_sa_eo_calib/`: 較正用。
- `data/results_sa_eo_states/`: `save_states=true` の分割ビット列付き run（EoFlip 系のみ対応）。

## 実行方法の要点
```
cargo build --release
./target/release/cli.exe --batch <batch.json> --out data/results_sa_eo --graphs data/graphs --threads 14
PYTHONUTF8=1 python experiments_sa_eo/quick_agg.py
```
バッチ JSON は `{"graphs": [...GraphSpec], "configs": [...RunConfig], "seed_start": k, "seed_count": 1, "save_states": false}`。
1 run の所要（10^7 ステップ、平均）: flipSA 1.2 s / swapSA 1.7 s / flipEO 8.8 s / swapEO 18.9 s。
