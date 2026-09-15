# 最終出力仕様：一次データを保存し、派生値は読み込み時に生成する

状態: 実装済みの保存仕様。例の数値とIDは説明用の架空データ。検証結果は[verification.md](verification.md)を参照。

## 1. 保存方針

永続化するのは、実験条件・グラフ・最終解と最良解・後から集計では復元できない計測値・実行時の所要時間と終了理由に限定する。表示用の情報や同じ条件を各結果へ複製しない。

| 情報 | 保存する一次データ | 読み込み・表示・export時に生成する値 |
|---|---|---|
| 実験条件 | バージョンを固定した未展開のExperimentSpecをバッチごとに1回 | 全ジョブ、条件ID、有効なK、単一ジョブの温度、近傍数、実効計測モード |
| グラフ | 頂点数と正規化した無向辺リストをグラフごとに1回 | 隣接リスト、辺数、次数、平均次数 |
| 結果の解 | 重複排除した分割プール、最終解・暫定解と各計測点のSolutionId参照、暫定解の初回到達ステップ | 実評価値、カット数、群サイズ、ペナルティ |
| 時系列 | 計測点の現行解・暫定解への参照、集計から復元できない平滑化・ベイスン測定値 | 参照分割から実評価と内訳を生成。平均・最小・比較表は外部で集計 |
| 実行情報 | 終了理由、完了ステップ、実測所要時間、必要な失敗情報 | 完了・失敗・未開始件数、結果ファイル一覧、進捗率 |
| 診断 | 有効化した場合のみ、再現できない実測カウンターと時間内訳 | 受理率、棄却数、時間比率 |

スキーマ・アルゴリズムのバージョン、チェックサム、試行識別子など、解釈・破損検出・書き込み整合性に必要な小さなメタデータは残す。再計算で得られる実験結果や大きな配列の重複とは区別する。

「集計から復元できる」と「探索やベイスン計測を再実行すれば計算できる」は区別する。各計測点の現行解・暫定解は共有分割プールへの参照として保存し、全探索履歴とベイスン終点の分割は保存しない。

## 2. 正本のファイル構成

```text
data/v1/
├── .writer.lock
├── graphs/<graph_id>.json
├── batches/<batch_id>/experiment.json
└── runs/<condition_id>/
    ├── seed_<seed>.json
    └── seed_<seed>.incomplete.json  # 最新の未完了試行がある場合だけ

export/                            # gpp exportを実行したときだけ作成
├── runs.tsv
├── traces.tsv
└── metadata.json
```

### experiment.json：条件の唯一の正本

schema_version、versions、specを保存する。specは入力時の既定値を明示した未展開のExperimentSpecで、グラフ生成条件、近傍方式、手法パラメータの配列、探索シードの配列、予算、計測設定を持つ。元のTOML/JSONそのものと別の展開済み設定を二重保存しない。

`budget.max_steps`はスカラーまたは正の整数配列。任意の`conditions[]`はグループごとの`neighborhoods`・`solvers`と、省略時に全体から継承する`budget`を持つ。グループを使う場合、ルートの近傍・ソルバー配列は空または省略する。予算配列は昇順、1要素ならスカラーへ正規化する。結果パスを計算する各展開条件の予算はスカラー一つとなる。

追加計測`measurement.best_basin`の既定値falseと空の`conditions`は保存時に省略する。この省略により既存のスカラー予算・追加計測なしの条件IDとバッチIDを維持する。trueは条件IDに含め、未計測の結果を計測済みとして再利用しない。ラウンド実行の有無・期限・スレッド数は実行オプションであり、保存条件やIDに含めない。

versionsはアルゴリズム、RNG導出、グラフ生成、計測、設定展開・正規化、および使用する適応度定義のバージョンを固定する。適応度の登録名はdefault、定義はgood_edge_fraction。名前とバージョンから決まる定義の説明文字列は結果へ複製しない。k_averageは設定として受け付けない。

入力のグラフサイズや近傍方式から有効なKなどを算出する。展開・正規化のバージョンが未対応なら再開を拒否する。展開済みジョブ一覧、設定ID一覧、job_keyから結果への対応表は保存しない。

### graphのJSON：グラフの唯一の正本

schema_version、node_count、edges、content_hashを保存する。edgesは頂点番号u<vのペアを辞書順に並べた配列。自己ループと重複は不可。content_hashはnode_countとedgesのチェックサムとして破損検出に使う。

生成条件とシードはexperiment.jsonから参照し、グラフJSONに重ねて書かない。node_countは孤立頂点の数を保持するために必要。幾何グラフの座標は初版では保存しない。グラフ構造に必要な辺リストを残し、座標が必要な場合は生成条件から再生成する。

### condition_idとseedの参照

condition_idは正規化した単一のグラフ生成条件、近傍方式、alpha、ソルバー・平滑化または適応度、予算、計測・診断条件、各バージョンから計算する。探索シードは含めず、ファイル名seed_<seed>.jsonに一度だけ置く。

実際のrun識別子は(condition_id, seed)の組。ディレクトリ・ファイル名に含まれるIDとシードは結果JSONの本文へ重複させない。各condition_idに対応する条件はexperiment.jsonを再展開して解決する。複数バッチで同一条件・シードになる場合は同じ結果を再利用する。

グラフIDは生成条件と生成器バージョンから計算し、グラフの内容は保存時のハッシュで検証する。同じIDのグラフを別の内容へ上書きしない。探索用のシード導出には実際のグラフ内容を用いるが、trajectory_id等の中間ハッシュは計算時だけ使用し、各結果に保存しない。

## 3. 完了結果JSONの保存項目

seed_<seed>.jsonは次の項目だけを持つ。

| キー | 内容 |
|---|---|
| schema_version | 結果形式の版。初版1 |
| attempt_id | 今回の試行識別子。未完了マーカーとの整合性確認に使う |
| termination | step_limit、local_optimum、no_sampled_improvement |
| completed_steps | 完了した探索ステップ数 |
| elapsed_ms | 単一実験の開始から結果組立までの実測時間。グラフ生成・ディスク保存を除く |
| partitions | 頂点番号順の真偽値配列を重複排除した配列。true=群A、false=群B |
| final_solution | `partitions`を参照するSolutionId。終了時の現行解 |
| best_solution | `partitions`を参照するSolutionId。終了時の暫定解 |
| best_step | 最良分割を最初に訪問したステップ |
| records | 次節で定義する疎な計測系列 |
| diagnostics | 有効化した場合だけ追加する最終累積診断値 |

完了専用ファイルなのでstatus=completedは保存しない。最終・最良スコア、カット数、群サイズ、ペナルティ、手法・設定・シードの本文複製、派生ID、終了件数、環境情報の大きなコピーは保存しない。初期分割は初期レコードからpartitionsを参照し、初期スコアはその分割から算出する。

各レコードの`current_solution`と`best_solution`も`partitions`のSolutionIdとする。同じ分割は同じIDを再利用する。最終・最良スコアは保存済みグラフと参照分割から算出する。最良値が同点なら最初の解を保持し、best_stepは後から復元できないため保存する。

## 4. records：復元できない計測だけを保存する

初期点0、指定された計測点、終了点を記録する。終了点が指定点と同じなら1件にまとめる。各レコードはstepと、その時点の必要な測定値だけを持ち、非適用・未実施のフィールドは省略する。

### 現在値・最良値

- 各計測点で`current_solution`と`best_solution`を保存する。実評価値は参照分割から算出し、current_real/best_realは保存しない。
- 初期点と終了点も通常のレコードとして保存し、終了点の参照はトップレベルのfinal_solution/best_solutionと一致させる。
- ステップ間に改善して再び悪化しても、計測点の現行解の最小値で暫定解を置き換えない。
- 計測点のカット数・群サイズ・ペナルティは保存せず、参照分割からRunViewとTSV生成時に算出する。

### 平滑化評価

- HC・SAで有効な平滑化がnone以外の場合、current_smoothedを保存する。
- ランダム平滑化では探索が保持した値と計測用固定サンプルの値が異なり得るため、search_evaluationも保存する。
- 決定的平滑化のsearch_evaluationはcurrent_smoothedと同じなので保存しない。
- noneでは平滑化関連の値を保存せず、必要な表示時にcurrent_solutionの実評価値から生成する。weighted_averageで有効K=0の場合も同じ扱いにする。
- EOには平滑化評価を保存しない。表示・exportでは非適用として空欄にする。

### ベイスン計測

要求した計測だけを次の小さなオブジェクトとして保存する。

| キー | 内容 |
|---|---|
| basin_real | 実目的関数で山登りした結果。real、terminationを保存。非自明な平滑化がある場合のみsmoothedも保存 |
| basin_smoothed | 平滑化で山登りした結果。real、smoothed、terminationを保存 |
| basin_best | `best_basin=true`の場合だけ、暫定解から実目的関数で山登りした結果。real、terminationを保存。smoothedは常に省略 |

terminationはlocal_optimumまたはstep_limit。値が局所最適か単なる打ち切り終点かを判別するために残す。走査数は通常保存せず、診断有効時だけstepsを追加する。

noneおよびweighted_averageの有効K=0では、both指定でもbasin_realだけ保存する。対応する平滑化ベイスンは表示・export時に同じ測定結果から生成する。EOではboth要求も実空間だけに解決し、basin_smoothedは非適用とする。有効な計測モードは条件から求まるので別フィールドとして保存しない。

ベイスン値は、現在スコアだけからは求められず、ベイスン終点の分割も保存しないため保持する。測定のために訪問した解を探索のbest_realへ混ぜない。

`basin_real`と`basin_smoothed`は現行解を始点とし、`measurement.basin`で選ぶ。`basin_best`は独立した選択なので`basin=none`でも保存できる。同じ暫定分割が続く計測点では直前の完了した`basin_best`を再利用する。専用の同点処理RNGはグラフ内容・近傍・alpha・探索シード・暫定分割・用途ラベルから導出し、計測ステップ・計算予算・ソルバーに依存させない。実評価値は始点の暫定解以下でなければならず、同じ暫定分割を参照する完了計測はビット単位で同じ値を持つ。終点の分割・群サイズ差は保存しない。診断のstepsは初回計算時の走査数であり、キャッシュ再利用時の追加計算量ではない。累積の目的関数実計算数には再利用を含めない。

暫定解ベイスンが有効な完了レコードでは`basin_best`を必須とする。中断時の最後の未完成計測は他の追加計測と同様に省略し、確定済みの現行解・暫定解参照だけを保存できる。未完成値をキャッシュに登録しない。

### 任意の診断情報

measurement.diagnosticsの既定値はfalse。trueの実行だけ、結果のdiagnosticsに以下の最終累積値を保存する。診断設定は結果条件の識別に含め、無効な結果を有効な結果として再利用しない。探索用RNGには影響させない。

- applied_moves（探索で適用した移動数。Swapは1移動）。
- objective_evaluations_search、objective_evaluations_measurement（実際に計算した目的関数値の数）。
- fitness_values_computed_search（EOのみ。キャッシュ参照を除く頂点適応度計算数）。
- search_ms、measurement_ms（時間内訳）。

各recordにカウンターや時間の累積値を繰り返し保存しない。accepted_movesは現在の全手法でapplied_movesと同じ、SAのrejected_movesはcompleted_steps-applied_movesなので保存しない。時間合計はelapsed_msを使い、other_msは必要時に差から算出する。無効時の診断値を0として扱わず、表示時は空欄とする。

## 5. 完了結果の具体例

保存先の例はruns/<condition_id>/seed_0.json。条件はexperiment.jsonに一度だけ置く。以下はEO、Swap、tau=1.5、fitness=default、最大3ステップ、診断なしの説明用結果であり、省略表記のない完全な結果JSON例。

グラフは4頂点のサイクル（辺0-1、1-2、2-3、3-0）。初期状態は[true,false,true,false]、最初の移動で[true,true,false,false]へ到達する。説明用の完全な結果であり、特定の生成シードの実測結果ではない。

```json
{
  "schema_version": 1,
  "attempt_id": "example-attempt",
  "termination": "step_limit",
  "completed_steps": 3,
  "elapsed_ms": 12.4,
  "partitions": [
    [true, false, true, false],
    [true, true, false, false]
  ],
  "final_solution": 1,
  "best_solution": 1,
  "best_step": 1,
  "records": [
    {
      "step": 0,
      "current_solution": 0,
      "best_solution": 0,
      "basin_real": { "real": 2.0, "termination": "local_optimum" },
      "basin_best": { "real": 2.0, "termination": "local_optimum" }
    },
    {
      "step": 1,
      "current_solution": 1,
      "best_solution": 1,
      "basin_real": { "real": 2.0, "termination": "local_optimum" },
      "basin_best": { "real": 2.0, "termination": "local_optimum" }
    },
    {
      "step": 3,
      "current_solution": 1,
      "best_solution": 1,
      "basin_real": { "real": 2.0, "termination": "local_optimum" },
      "basin_best": { "real": 2.0, "termination": "local_optimum" }
    }
  ]
}
```

この例の計測設定はexplicitのsteps=[1]、basin=real、best_basin=true。0と終了点3を自動追加する。step=0のベイスン値は2だが、探索の現行解・暫定解はともにpartitions[0]から復元する。終了点の現行解・暫定解はpartitions[1]を共有参照し、step=3のbasin_bestはstep=1の完了計測を再利用する。

## 6. 再開・失敗を扱う最小の運用データ

plan.json、manifest.json、summary.json、展開済みジョブ一覧、状態一覧の正本は作らない。inspect・resumeはexperiment.jsonを再展開し、期待する結果パスを求め、ファイルを検証して状態と件数をその都度計算する。BatchSummaryはメモリ上の戻り値・標準出力として維持し、自動保存しない。

ラウンドも保存シードの数値昇順から再構成する。`completed_rounds`と`deadline_reached`はBatchSummaryだけに含める。期限による停止では未開始ジョブのマーカーを作らず、実行中のラウンドの確定を待つ。再開時は既存の完了結果を再利用して不足分を実行するため、ラウンド用マニフェストやタイマー状態を保存しない。

ジョブ開始時にseed_<seed>.incomplete.jsonを小さなマーカーとして作る。内容はschema_version、attempt_id、status=running。協調的中断または失敗時に、status=cancelled/failed、error（code/messageと必要最小限の文脈）、保存可能なpartial_resultへ更新する。条件や完了結果は複製しない。

partial_resultは完了結果の解・ステップ・records・時間と同じ構造を用い、状態とattempt_idは親から継承して重複させない。解を初期化できなかった場合はpartial_resultを省略する。中断で未完了になった最後の計測は測定値を作らず、未実施として扱う。途中状態から再開するためのRNG状態は保存しない。

成功時は同じattempt_idの完了結果を原子的に確定してマーカーを削除する。結果保存直後に停止しマーカーだけ残った場合、attempt_idが一致する有効な完了結果を優先して残ったマーカーを清掃する。排他ロックを取得した再開処理でrunningマーカーだけが残っていれば中断と判断する。

以前の完了結果と異なるattempt_idの失敗マーカーが共存する場合、上書き再実行が失敗した状態。旧結果は維持し、inspectで有効な完了結果と最新失敗を区別する。通常再開は有効な旧結果を再利用し、再計算は--overwriteで指定する。

有効な完了結果と壊れた、または未対応の将来版の未完了マーカーが併存するときは、完了結果を正としてinspect・export・再利用を続ける。マーカーは削除せず、inspectでは`latest_attempt_status=invalid_marker`および`error.code=marker_issue`として別報告する。完了結果がない壊れたマーカーは通常のrunでは拒否する。CLIの`resume`、`--overwrite`、またはRustの`recover_corrupt=true`では、一意な調査用退避ファイルへコピーして同じseedを先頭から再計算する。未来版の未完了マーカーと完了結果は自動変換も上書きも禁止する。読み取り・結果確定に必要なI/Oのエラーは破損復旧として扱わず伝播する。結果確定後のマーカー清掃失敗は完了結果を失敗へ戻さず、次の書き込み時に清掃を再試行する。

成功履歴・すべての過去試行は保管しない。通常は最新の未完了情報だけを保持し、新しい試行で置き換える。破損結果または破損マーカーを明示的に復旧する場合だけ、調査用の退避ファイルを残す。結果の保存失敗時に完了を通知しない。

同じ保存ルートへの書き込みはOSが解放する排他ロックで一つに制限する。永続的なrunning一覧やPID一覧を別管理しない。グラフと結果の一時ファイルは未確定として無視する。

## 7. export時に生成するもの

exportは条件・グラフ・疎な結果を読み、便利な表へ展開する。TSVは分析用の生成物であり、通常実行は作成しない。いつでも削除・再生成できる。平均・標準偏差・手法比較や作図そのものは外部ツールで行う。

### TSV列の唯一の定義

export実装の`ColumnSpec`が、`runs.tsv`と`traces.tsv`の列順、型、単位、空欄規則、`metadata.json`を生成する唯一の定義である。以下は利用者向けの固定契約であり、ヘッダーやmetadataを別のリストから組み立てない。

### runs.tsv：1ジョブ1行

列順は以下の順で固定する。

1. batch_id、condition_id、seed、status、latest_attempt_status、termination。
2. graph_id、graph_kind、node_count、expected_degree、graph_seed、edge_count、actual_average_degree、alpha、neighborhood。
3. solver、temperature、tau、smoothing、k、fitness、fitness_version、fitness_params_json。
4. max_steps、completed_steps、best_step、initial_real、final_real、best_real、final_cut_edges、final_size_a、final_size_b、final_balance_penalty、best_cut_edges、best_size_a、best_size_b、best_balance_penalty、elapsed_ms。
5. final_basin_real_from_real、final_basin_real_status、final_basin_real_from_smoothed、final_basin_smoothed_status、final_basin_real_from_best、final_basin_best_status。
6. applied_moves、accepted_moves、rejected_moves、objective_evaluations_search、objective_evaluations_measurement、fitness_values_computed_search、search_ms、measurement_ms。

条件はexperiment.jsonの展開結果から、グラフの統計は辺リストから、初期・最終・最良の評価内訳は参照分割から算出する。completed_steps=0の部分結果も確定した分割から算出する。未開始・失敗ジョブも条件と状態の行は出し、結果のない列は空欄。診断無効時は診断由来列も空欄。通常の正本にこの表を保存しない。

### traces.tsv：1計測点1行

列順はcondition_id、seed、status、step、current_real、best_real、search_evaluation、current_smoothed、basin_real_from_real、basin_smoothed_from_real、basin_real_status、basin_real_steps、basin_real_from_smoothed、basin_smoothed_from_smoothed、basin_smoothed_status、basin_smoothed_steps、basin_real_from_best、basin_best_status、basin_best_steps。実評価値はJSONのcurrent_solutionとbest_solution参照から生成する。

初期・終了点の実評価値、noneの場合の平滑化値、重複を省略したベイスン値は、前述の規則で補完する。分割配列とSolutionIdはTSVへ複製しない。保存された途中計測点のカット数・群サイズ・ペナルティはJSONの参照分割から取得できる。途中の時間・カウンターは元データがないため追加しない。EOの平滑化列は空欄。未計測や診断無効のstepsも空欄。condition_idとseedでruns.tsvへ結合できる。

出力例の主要列だけを抜き出すと次の表になる。

| step | current_real | best_real | basin_real_from_real | basin_real_status | basin_real_from_best | basin_best_status |
|---|---|---|---|---|---|---|
| 0 | 4 | 4 | 2 | local_optimum | 2 | local_optimum |
| 1 | 2 | 2 | 2 | local_optimum | 2 | local_optimum |
| 3 | 2 | 2 | 2 | local_optimum | 2 | local_optimum |

--include-incomplete指定時だけ、完了結果がないジョブの最新の読み取り可能な部分結果を追加する。異なる試行の系列を連結しない。有効な完了結果がある場合はそれを出力する。

### metadata.json

形式バージョン、出力時刻、対象バッチ、列の名前・型・単位・意味、部分結果を含むかをexport時に生成する。単一実行への絞り込みがある場合はその条件IDとシードも含める。条件全文、展開済み全ジョブ、結果の件数集計やコピーは入れない。処理件数や除外理由の要約は標準出力へ表示する。

列名を先頭行に置き、コメント行は作らない。空値は空欄、数値は往復可能な精度で出力する。文字列のタブ・改行・引用符は引用処理する。行順はcondition_id、seed、stepの順。既存出力は--overwriteなしで上書きしない。存在するはずの結果が破損していれば非0で終了し、黙って除外しない。

## 8. 保存削減による制約と検証

通常保存しないものは、各ベイスン終点の分割、全頂点適応度の時系列、全探索・受理履歴、RNG状態、展開済み計画、状態一覧、集計表、TSV、幾何座標。初期・計測点・終了点の現行解と暫定解はpartitionsとSolutionIdで保存する。全頂点適応度の定義・パラメータはexperiment.jsonから参照する。

初期分割は初期化の再実行で復元できる。途中分割や省略した未計測点のベイスン値は集計では復元できず、元の探索・計測の再実行が必要。保存した分割から復元できる内訳と、復元できない途中の内訳を混同しない。

実装テストには次を含める。

- 正本に展開済み計画、状態集計、設定コピー、初期分割、解から導出できる終点スコアが含まれないこと。
- 初期・終了点のcurrent_solution/best_solution参照、分割プールの重複排除、途中の暫定解の保持。
- none/weighted K=0の重複値補完、EOの非適用、計測なし、打ち切り、診断無効を区別できること。
- 診断の有効・無効で探索結果が変わらず、診断無効時に値を0として偽装しないこと。
- manifestやsummaryなしで完了・未開始・中断・失敗を判定し、再開できること。
- 上書き失敗時の既存完了結果の維持と、残存マーカーの処理。
- 生成したTSVを削除して再生成し、時刻以外が同じになること。実験の再実行を必要としないこと。
