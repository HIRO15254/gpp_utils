# 旧実験履歴の新形式（data/v1）への変換記録

対象: 2026-09-16 に取り込んだ上流リファクタ（`51577f9`..`7b37ef1`）で保存形式が
変わったあとの、既存実験データの扱い。実際に変換したもの・変換できないもの・その
判定根拠を残す。変換の実行は `experiments_sa_eo/convert_to_v1.py`。

```
PYTHONUTF8=1 python experiments_sa_eo/convert_to_v1.py all
```

## 0. 2026-09-25 の更新（アルゴリズム v2）

実装変更（`multiplicative`・`additive` の組み込み適応度、EO 選択規則 v2、tau >= 0、結果 JSON の分割表現の圧縮、
serde_json の `float_roundtrip`）に合わせて、次の点がこの記録の元の記述から変わった。以下の本文は 2026-09-16
時点の判断記録として残し、現行の状態はこの節を正とする。

- **同日、ベースラインを再定義した。** 今後の計算はこれを基準とする（`AGENTS.md`「実験のベースライン」）。
  - 新しい条件: Θ −1.50〜+2.50 と τ 0.00〜3.00（ともに 0.05 刻み）、18 組 × インスタンス 1 つ（s0）、
    実行シード 32、10^6 ステップ。284 条件/グラフで **163,584 ジョブ**。
  - `baseline_v1.toml` はこの新定義で、`experiments_sa_eo/make_baseline_v1.py` が生成する。
  - `convert_to_v1.py` の格子は旧ベースラインのまま残した。用途はグラフ変換と `condition_map_v1.csv` だけで、
    `spec` サブコマンドは廃止した（新しい `baseline_v1.toml` を旧格子で上書きしないため）。
- tau = 0.00 が表せるようになった。旧格子（196 条件/グラフ、451,584 ジョブ。旧 `data/results_sa_eo`
  484,842 run のうち 93.1%）はすべて新形式で表せる。`condition_map_v1.csv`（14,112 行）はこの旧格子の対応表。
- `pinned_versions.algorithm` が v2 になり、全条件 ID が変わった。`condition_map_v1.csv` と
  `baseline_v1.toml` は v2 の ID で作り直した（`convert_to_v1.py spec` / `map`）。
- 旧 `EoFlipMulAlpha{alpha}`・`EoFlipAddBeta{beta}` は、それぞれ組み込みの `multiplicative{alpha}`・
  `additive{beta}` と同じ式・同じ多数派判定・同じ同点平均化規則なので、**条件としては表せる**ようになった。
  適応度スクリーニング（`data/results_sa_eo_screen`）と途中停止したパイロット（`data/results_sa_eo` 内の
  33,258 run）の条件も新形式で作り直せる。旧 run 自体を変換できない理由（§4: 計測点ごとの分割が無い、
  乱数導出が違う）は変わらない。`MulGamma`・`Legacy`（仮想頂点型）は引き続き組み込みに無い。
- Swap の 2 頂点目は、旧実装の「反対側が出るまで最大 50 回再抽選し、失敗したら一様」ではなく、
  平均化した重みでの厳密な条件付き分布から引く。Flip の選択分布は旧実装の最終規則と同じ。
- §5 の件数訂正: `data/results_rankdiv` は 36,864 run（+ 状態ファイル 36,864。元の表の 73,728 は
  状態ファイルを数えていた）。パイロット 3 ストア（`results_pilot_a/_b`、`results_rankdiv_pilot`）は
  計 26 run（元の 51 は状態ファイル込み）。

## 1. 結論

| 対象 | 判定 | 実施 |
|---|---|---|
| グラフ 72 本（`data/graphs/*.json`） | 変換可（完全一致） | 済み → `data/v1/graphs/` |
| ベースライン格子の条件定義 | 変換可（194/196 条件） | 済み → `experiments_sa_eo/baseline_v1.toml` |
| 旧条件 ↔ 新 condition_id の対応 | 算出可 | 済み → `experiments_sa_eo/condition_map_v1.csv` |
| 旧 run 結果（`seed_<s>.json`） | **変換不可** | 見送り（§4） |
| 適応度スクリーニング系ストア | **条件が表せない** | 見送り（§5） |

旧 run の中身を新形式の `runs/<condition_id>/seed_<s>.json` にすることはできない。
理由は 2 つあり、どちらも単独で致命的（§4）。旧データは旧形式のまま
`experiments_sa_eo/*.py` の集計系で使い続けるのが正しい。

## 2. グラフ（変換済み・完全一致を確認）

旧形式は `spec` + `adjacency_list` + `coordinates` + `edge_count`、新形式は
`schema_version` + `node_count` + `edges`（u<v の辞書順）+ `content_hash`。
隣接リストから辺集合を正規化し、`Graph::content_hash()` と同じ
SHA-256（`"gpp-graph-v1\0"` + node_count + 辺列、すべてリトルエンディアン 8 バイト）
を計算して書き出す。幾何グラフの座標は新仕様では保存しない（必要なら生成条件から
再生成する）。

確認したこと:

- 新実装 `Graph::generate()` で 72 本すべてを生成し、旧 `data/graphs/*.json` の辺集合と
  **72/72 完全一致**。乱数は旧実装と同じ `Mt19937GenRand64::new(spec.seed)` で、生成
  ループ順も同じ。幾何グラフの辺条件は旧 `sqrt(dx^2+dy^2) <= sqrt(d/(n*pi))` が新
  `dx^2+dy^2 <= d/(n*pi)` になったが、この 72 本では判定が変わらなかった。
- 変換して書き出した 72 ファイルは、`gpp` 自身が生成したファイルと**バイト一致**
  （`serde_json::to_vec_pretty` + 末尾改行に合わせてある）。

つまり問題例は新旧で同一であり、新形式で回した結果を旧結果と「同じインスタンス上の
値」として比較できる。

## 3. 条件定義（変換済み・194/196）

旧 4 系列はすべて新形式の (neighborhood, solver) で表せる。

| 旧 `solver` | 新 `neighborhood` | 新 `solver` | 備考 |
|---|---|---|---|
| `"Sa"` | `flip` | `sa { temperature, smoothing = none }` | T = 10^Theta |
| `"SaSwap"` | `swap` | `sa { temperature, smoothing = none }` | 同上 |
| `{"Eo": {tau}}` | `swap` | `eo { tau, fitness = "default" }` | `default` = `good_edge_fraction-v1` = g/deg |
| `{"EoFlipMulAlpha": {tau, alpha: 1.0}}` | `flip` | `eo { tau, fitness = "default" }` | alpha=1 で lambda1 が常に 1（旧 `eo_flip_lambda_mul_alpha` は `swap_fitness * 1.0`）なので g/deg と完全一致 |

計測点も一致する。新 `schedule = "logarithmic"` の生成列（0, 1..9, 10..90, …, 10^7）は
旧実装の 65 点と同じ。旧 records の `basin_real_from_real` / `basin_real_from_best` は
新 `measurement.basin = "real"` + `best_basin = true` に対応する。

表せない 2 条件（各 EO 系列の tau = 0.00）:

- 新実装は tau > 0 を要求する（`src/experiment/plan.rs`: `tau must be finite and positive`）。
- 旧格子の tau = 0.00 は一様ランダム選択の極限として入れていたもの。swapEO・flipEO
  各 1 点、72 グラフぶんで 144 条件 = 2,304 + 2,304 run が対象外。

結果として `baseline_v1.toml` は 72 グラフ × 194 条件 × 実行シード 32 = **446,976 ジョブ**。
旧 `data/results_sa_eo` の全 484,842 run のうち **446,976 run（92.2%）**が、条件としては
1 対 1 で新形式に対応づく（対応表は `condition_map_v1.csv`、13,968 行）。対応表の
`old_condition_dir` 13,968 件はすべて旧ストアに実在することを確認した。

較正用ストア `data/results_sa_eo_calib` は同じ 4 系列の 10^6 ステップ版なので、
`baseline_v1.toml` の `max_steps` を `1000000` にすれば同じ条件集合が作れる。

## 4. 旧 run 結果は新形式にできない

### 4.1 新スキーマが要求する情報が旧ファイルに無い

新 `MeasurementRecord` は各計測点で `current_solution` / `best_solution`（`partitions`
プールへの `SolutionId`）を**必須**で持ち、実評価値はそこから毎回算出する
（`docs/output-format.md` §3–4、`RunResult::validate`）。旧 run ファイルが持っているのは

- `final_partition`（終了時の分割のみ）
- `records[]` のスカラー値（`current_real`, `best_real`, `basin_*`）

であって、**各計測点の分割・暫定解の分割・`best_step` は保存していない**。スカラーから
分割を復元することはできない（同じ評価値を与える分割は多数ある）。65 点ぶんの
`SolutionId` を埋める術がないので、新形式の run JSON は作れない。捏造した分割を入れれば
`validate` は通るが、参照分割から再計算される実評価値が実際の探索値と食い違い、
データとして壊れる。

例外候補だった `data/results_sa_eo_states`（`save_states=true`、65 スナップショットの
分割ビット列つき、848 run）も、(a) 現行解のみで暫定解の分割が無い、(b) 条件が
`EoFlipMulAlpha` alpha=1 以外を含む、の 2 点で足りない。

### 4.2 乱数導出仕様が変わったので混ぜてはいけない

- 旧: 実行シードを `Mt19937GenRand64::new(seed)` へ直接投入。
- 新: `rng: sha256-mt19937-64-v1`。グラフ内容・近傍・alpha・用途ラベル等から
  SHA-256 で 64bit シードを導出（`src/optimization/rng.rs`）。

`docs/verification.md` にも「乱数導出仕様は変更しており、同一探索軌跡の速度比較では
ない」と明記されている。実測でも確認した（Random n=124 d=5 s0、flipSA Theta=0、
実行シード 0、10^7 ステップ）:

| | 初期値 | 最終 current | best | basin(現行解) |
|---|---|---|---|---|
| 旧 `data/results_sa_eo` | 162.2 | 96.2 | 64.2 | 74.0 |
| 新実装 | 156.8 | 93.8 | 62.8 | 69.0 |

初期分割からして別物であり、同じ (condition, seed) でも別軌跡になる。仮に旧結果を
`runs/<condition_id>/seed_<s>.json` に置くと、`gpp run` / `resume` が「既存の完了結果」
として黙って再利用し、新旧の軌跡が 1 条件の中で混ざる。過去に同種の取り違え
（iter6/iter8 のデータ混入）で結論を取り違えた経緯があるため、これは明示的に避ける。

`condition_map_v1.csv` は「旧結果を新 run として流用する」ためのものではなく、
**同じ条件の旧結果と新結果を突き合わせる**ためのものとして使う。

## 5. 条件自体が表せないストア

新実装の適応度レジストリには `default`（`good_edge_fraction-v1` = g/deg）しか無い。
追加の組み込み適応度は上流で意図的に除外された（`docs/verification.md`「除外を確定した
範囲」）。カスタム適応度は `FitnessRegistry::register` で呼び出し側に登録できるが
（`examples/custom_fitness.rs`）、それはコード追加であってデータ変換ではない。

| ストア | run 数 | 状態 |
|---|---|---|
| `data/results_sa_eo_screen` | 396,288 | `EoFlipAddBeta`（lambda = beta*lambda0 + lambda1）。要カスタム適応度 |
| `data/results_rankdiv` | 73,728 | 同上 |
| `data/results_sa_eo_states` | 848 (+ 848 states) | `EoFlipMulAlpha` alpha=1。条件は表せるが §4.1 で run 不可 |
| `data/results_pilot_a` / `_b`, `data/results_rankdiv_pilot` | 51 | `EoFlipMulGamma` / `EoFlip` legacy。要カスタム適応度 |
| `data/results_sa_eo` の一部 | 37,866 | tau=0（§3）と、このストアに混在している AddBeta・MulAlpha alpha≠1 のパイロット条件 |
| `data/results`（iter6 世代） | 168,004 | 平滑化スイープが主。`KAverage(k)` は上流で削除、旧 `WeightedAverage(w: f64)` は新 `weighted_average { k: usize }` と意味が違う。`None` と `RandomKAverage(k)` の部分だけが対応する |

## 6. 生成物

| パス | 内容 |
|---|---|
| `data/v1/graphs/*.json` | 変換済みグラフ 72 本 |
| `experiments_sa_eo/baseline_v1.toml` | ベースライン格子の新形式実験仕様（446,976 ジョブ） |
| `experiments_sa_eo/condition_map_v1.csv` | 旧条件ディレクトリ → 新 condition_id（13,968 行） |
| `experiments_sa_eo/convert_to_v1.py` | 上記を生成するスクリプト。格子定義（Theta/tau/グラフ/シード）の単一の真実の源も兼ねる（旧 `grid.py` は上流で削除された） |

`baseline_v1.toml` を実行するとベースライン全体を新実装で回すことになる。旧実装の
実測（flipSA 1.2s / swapSA 1.7s / flipEO 8.8s / swapEO 18.9s per run）から、
446,976 run は総計で数千 CPU 時間の規模なので、回すかどうかは別途判断する。
