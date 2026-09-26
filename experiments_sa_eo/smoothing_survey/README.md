# 解空間平滑化の調査：再現用スクリプト

[docs/smoothing-survey.md](../../docs/smoothing-survey.md)の数値確認と予備実験を再現するためのスクリプトである。

- ここにあるものは`gpp`本体ではない。CI（`scripts/check.py`）の対象外で、`gpp`の計算結果にも影響しない。
- 予備実験はベースラインの正式な比較ではない。グラフはベースラインと同じだが、シード・Θ・計測が異なる。正式な比較は`gpp`へ実装してから、ベースライン条件で行う（本文第6章）。

## theory/：命題の数値確認（Python 3、numpy、scipy）

| スクリプト | 内容 | 所要時間の目安 |
|---|---|---|
| `check_affine.py` | 命題1（Flip・Swapの全近傍平均、Flipの距離2球面）と命題2（Gu–Huangの退化）を全列挙で確認 | 数十秒 |
| `check_affine2.py` | 命題1の補足（半径3の球面、独立反転ノイズ、Swapの距離2クラス） | 1分程度 |
| `check_pseudo_marginal.py [steps]` | 命題3。`random_k_average`付きSAの経験分布をpseudo-marginalの理論分布と比べる | 約45秒（2×10^6ステップ） |
| `hk_landscape.py [n]` | 提案B。熱核平滑化の厳密局所最小解・最適解の保存・継続の成功率 | n=16で約10秒、n=20で約2分 |
| `landscape2.py swap\|flip n` | 台地込みの局所最小解の数（Swap：熱核、Flip：近傍ソフトミン）と最急降下の成功率 | n=16で数十秒 |
| `check_erosion.py` | 命題4(6)。エロージョン地形の局所最小解の読み出しと、距離2〜4の局所最小解の間の障壁 | 1分程度 |
| `check_swap_softmin.py` | 提案AのSwap版で、交換近傍のボルツマン和の式（群ごとの和の積とカット辺の補正）を全等分割で確認 | 1分程度 |

```text
python experiments_sa_eo/smoothing_survey/theory/check_affine.py
python experiments_sa_eo/smoothing_survey/theory/check_pseudo_marginal.py 2000000
python experiments_sa_eo/smoothing_survey/theory/landscape2.py flip 16
```

## pilot/：予備実験のプロトタイプ（Rust、Python）

`smoothing_pilot`は、Flip近傍の定温MAに次の平滑化を実装した単独のクレートである。

- `none`：平滑化なし
- `rk1`・`rk4`：`gpp`の`random_k_average`と同じpseudo-marginal評価（K=1、4）
- `nsm_rho1`・`nsm_rho0.5`・`nsm_erosion`：提案Aの近傍ソフトミン（λ=ρT、κ=1。erosionはλ→0）
- `hk_tau0.5`・`hk_tau2`・`hk_sched`：提案Bの熱核平滑化（静的τ=0.5、2。スケジュールはτ=4から半減して最後の2×10^5ステップを元の問題にする）

グラフ生成は`gpp`の`Graph::generate`と同じ式・乱数である。ベースラインの18グラフが`gpp`の保存グラフと辺リストまで一致することを、`gpp_graphs.toml`（グラフを生成させるだけの1ステップの仕様）と`compare_graphs.py`で確認した。探索の乱数系列は`gpp`とは異なる。計測は0、10^3、10^4、10^5、10^6ステップの暫定解（実評価が最良の現行解）と、そこからの最急降下（同点は番号の小さい頂点）によるベイスン値である。

```text
cd experiments_sa_eo/smoothing_survey/pilot
cargo build --release --locked
./target/release/smoothing_pilot selftest                 # 全近傍の直接計算との一致を確認
./target/release/smoothing_pilot graphs ../out/graphs      # ベースラインの18グラフを書き出す
gpp run gpp_graphs.toml --root ../out/gpp                 # 同じグラフをgppで生成（確認用）
python compare_graphs.py ../out/gpp/graphs ../out/graphs  # 辺リストの一致を確認
python eig.py ../out/graphs ../out/eig                     # ラプラシアンの固有分解
./target/release/smoothing_pilot run ../out/eig all all -100 120 10 8 > ../out/pilot.tsv
python analyze.py ../out/pilot.tsv 1000000                 # 交差検証した最適Θでの比較表
python analyze.py ../out/pilot.tsv --summary pilot_summary.tsv
```

`run`の引数は、固有分解のディレクトリ、グラフ名の部分一致（カンマ区切り、`all`で全部）、手法（カンマ区切り、`all`で全部）、Θ×100の下限・上限・刻み、シード数の順である。本文の予備実験は上の`run`の行（Θ=-1.0〜+1.2の0.1刻み23点、シード0〜7）で、4スレッドで約1時間かかった。`pilot_summary.tsv`は、その結果をグラフ×手法×Θごとに集計したものである。

## 本文との対応

| 本文 | 確認に使ったもの |
|---|---|
| 命題1・系の表 | `check_affine.py`、`check_affine2.py` |
| 命題2 | `check_affine.py` |
| 命題3の表 | `check_pseudo_marginal.py` |
| 命題4（全列挙の数値） | `check_erosion.py`、`landscape2.py flip 16` |
| 第5章 | `pilot/`一式と`pilot_summary.tsv` |
