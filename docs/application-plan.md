# gpp_utils アプリケーション仕様

この文書は、現在提供するローカル実験アプリケーションの範囲、CLI・設定、責務境界、公開 API を定める。計算規則は[algorithms.md](algorithms.md)、保存形式とTSV列は[output-format.md](output-format.md)、計算結果を維持する高速化の比較契約は[performance.md](performance.md)、過去の実装・測定の証跡は[verification.md](verification.md)を正本とする。履歴上のテスト件数や測定値は、現在のCIの成功を意味しない。

## 1. 目的と対象範囲

完成版は、グラフ分割問題について、条件を定義し、複数手法・パラメータ・シードで実験し、中断や失敗から再実行し、結果をJSON・TSVとして取り出せるローカルCLIアプリケーションとRustライブラリとする。

今回、従来保留していたTOML設定、パラメータスイープ、HC・EOのバッチ対応、探索と計測の乱数分離、実行中キャンセル、未完了ジョブの再実行を完成させる。GUIとSQAは本体・公開API・依存・テスト・使用例・現行の説明から削除する。

ユーザーが確定した方針:

- GUIとSQAは不要。
- 全手法でFlipと等分割Swapを選択できるようにする。EOのtausは必須。
- EOの組み込み適応度は`default`・`multiplicative`・`additive`の3種（定義は`good_edge_fraction-v1`・`multiplicative-v1`・`additive-v1`）。Rustで定義を追加・登録し、設定から選択する拡張性は維持する。
- 平滑化はHC・SAのみ。`k_average`は削除する。
- EOの提案とSAのMetropolis判定を組み合わせたEO-SA（`eo_sa`）を追加する。平滑化は扱わない（2026-09-26）。
- 旧Rust API、CLI、設定、保存形式との互換性は不要。既存データ自体は削除しない。
- 完成形の構成と実際に使える機能をこの文書にまとめ、実装時も更新する。

追加回答で確定した方針と、本計画の実装範囲:

- 連続緩和は対象外とすることが確認済み。現状は未完成ランナーからの呼び出しのみで、連続目的関数・離散化の実装がない。完成版の設定やコマンドにも未実装の選択肢を残さない。
- JSON・TSV出力のみとし、集計・作図を外部で行うことが確認済み。シード平均、統計比較、gnuplot、グラフ描画、Web画面は含めない。実験の状態や各実行の最良値を一覧にする機能は含める。
- 再開はジョブ単位。完了済みジョブを再利用し、未完了ジョブを同じ条件・シードで最初から実行する。探索の途中状態やRNG状態を復元するチェックポイント再開は実装しない。
- 単一マシンでのCPU並列実行を対象とし、分散実行・サーバー・クラウド連携は含めない。

## 2. 対象の問題と探索機能

無向・重みなしのRandom／Geometricグラフを2群へ分割し、カット数と群サイズ差のペナルティを最小化する。Flipは任意分割、Swapは偶数頂点の等分割を扱う。HC、固定温度SA、EO、EO-SAを提供し、HC・SAにはnone／all_average／random_k_average／weighted_averageを組み合わせる。EOのtauは必須（`tau >= 0`）で、EO-SAはtausと温度（`temperature >= 0`）の両方が必須。組み込み適応度はdefault・multiplicative・additiveの3種。独自適応度はRustで登録する。

目的関数、受理・同点規則、EO順位抽選、孤立頂点、平滑化Kと距離2の境界・正規化、差分評価・生成式の正本は[algorithms.md](algorithms.md)。設定検証はその規則に従い、第4節の形式を展開する。ここでは計算仕様を重複管理しない。

## 3. ユーザーの操作とCLI

Cargoパッケージとライブラリ名は`gpp_utils`、配布する実行ファイルは`gpp`とする。旧`cli`・`gui`・`experiment`バイナリは廃止し、入口を一つにする。

| コマンド | 実際にできること |
|---|---|
| `gpp init experiment.toml` | コメント付きの小規模設定例を生成。`.json`なら同等のJSONを生成。既存ファイルは上書きしない |
| `gpp validate experiment.toml` | 構文、手法固有のパラメータ、全組み合わせ、重複、数値範囲を検証。グラフ生成・実験はしない |
| `gpp plan experiment.toml` | 展開した条件と総ジョブ数を表示。`--out experiment.json`指定時だけ、既定値・バージョンを固定した未展開の条件を保存。グラフ生成・実験はしない |
| `gpp run experiment.toml --root data/v1 --threads 4` | 検証、未展開の条件保存、グラフ準備、並列実行、結果保存を行う |
| `gpp run --experiment experiment.json --root data/v1 --threads 4` | 保存したバージョン付き条件を展開して実行。設定ファイル入力との同時指定は不可 |
| `gpp resume --batch <batch_id> --root data/v1 --threads 4` | 保存条件を再展開し、検証済み完了結果を再利用して残りを最初から実行 |
| `gpp inspect --batch <batch_id> --root data/v1` | 計画、状態別件数、各ジョブの実スコア最良値、終了理由、エラーを一覧表示 |
| `gpp inspect --batch <batch_id> --condition <condition_id> --seed <seed>` | 単一実行の条件、分割から算出した最終・最良指標、計測状態を表示 |
| `gpp export --batch <batch_id> --root data/v1 --out export` | 条件と結果からTSVを生成。`--condition <condition_id> --seed <seed>`で単一実行に限定可 |

`run`と`resume`は既定で利用可能な論理CPU数をワーカー数とする。`--threads`は1以上とし、実験の科学的な条件や識別子には含めない。`--root`の既定値は現在の作業ディレクトリを基準とする`data/v1`。設定内に出力先やスレッド数を重複して持たせない。

`run`・`resume`の`--rounds`は、探索シードの数値昇順で1シードを1ラウンドとして実行する。同じラウンドの全条件を並列実行し、全ジョブが完了または再利用された後で次へ進む。失敗・中断・未開始が残るラウンドから先へは進まない。再開時は有効な結果を再利用し、先のラウンドの不足分から埋める。全件再利用のラウンドも`completed_rounds`に数える。

`--deadline-seconds <秒数>`は`--rounds`と同時に指定する実行時間の目安で、各ラウンド開始前だけに確認する。実行中のラウンドを期限で打ち切らないため、指定時間を超える場合がある。0はジョブを開始せず停止する。期限到達は`deadline_reached=true`・終了コード0、Ctrl+Cは期限より優先して130とする。ラウンド・期限・スレッド数は実行オプションであり、条件IDや保存設定に含めず、進捗ファイルも追加しない。

`run --overwrite`は選択された計画のジョブだけを再計算する。再計算失敗時に既存の完了結果は失わない。指定なしでは、検証済みの同一結果があれば再利用する。全出力先を消去する操作は用意しない。

`validate`・`plan`・`inspect`・`run`・`resume`の`--json`で機械可読な最終結果を標準出力へ出せるようにする。進捗と診断は標準エラー、最終結果は標準出力に分け、ライブラリ自体はprintしない。

終了コードは0=成功またはラウンド期限による停止、1=実行・保存・出力などの失敗、2=引数・設定エラー、130=ユーザー中断とする。スキップのみの実行も成功。バッチ内に失敗ジョブがある場合は実行対象の他ジョブを継続し、最終終了コードを1とする。ラウンド実行では同じラウンドまで継続し、後続ラウンドは開始しない。中断と失敗が混在する場合は130を返し、失敗件数は標準出力の最終報告に含める。

典型的な操作:

```text
cargo build --release --bin gpp
gpp init experiment.toml
gpp validate experiment.toml
gpp plan experiment.toml
gpp run experiment.toml --threads 4
gpp inspect --batch <runコマンドが表示したbatch_id>
gpp resume --batch <batch_id> --threads 4
gpp export --batch <batch_id> --out export
```

## 4. 設定とスイープ

TOMLとJSONは同じ`ExperimentSpec`へ読み込む。形式による動作差はなく、拡張子で選択する。未知のフィールドと非対応の手法をエラーにし、誤記を無視しない。

設定は次の構成とする。

| 項目 | 意味・既定値 |
|---|---|
| `schema_version` | 必須、初版は1 |
| `name` | 任意の表示名。省略時は入力ファイル名。識別子には含めない |
| `run_seeds` | 必須、空でないu64の配列。探索を繰り返すシード |
| `neighborhoods` | `conditions`を使わない場合に必須。flip・swapから選ぶ非空・重複なしの配列 |
| `problem.alpha` | 有限かつ0以上。既定0.05 |
| `budget.max_steps` | 必須、1以上の整数または非空・重複なしの整数配列。全条件の既定予算 |
| `measurement.schedule` | `logarithmic`（既定）または`explicit` |
| `measurement.steps` | `explicit`のときのみ指定する一意なステップ配列。0と終了時点は自動追加 |
| `measurement.basin` | `none`、`real`、`both`。既定`both` |
| `measurement.best_basin` | 既定false。暫定解から実目的関数のベイスンを計測。`basin`とは独立 |
| `measurement.max_basin_steps` | 各ベイスン山登りの最大走査数。1以上、既定10,000 |
| `measurement.diagnostics` | 既定false。trueの場合のみ最終累積カウンター・時間内訳とベイスン走査数を保存 |
| `graphs[]` | `kind`、`node_counts[]`、`expected_degrees[]`、`seeds[]`。各配列は必須・非空 |
| `solvers[]` | `conditions`を使わない場合に必須。`kind`と手法固有パラメータ配列、`smoothing[]` |
| `conditions[]` | 任意。各グループの非空`neighborhoods`・`solvers`と任意の`budget`。使用時はルートの`neighborhoods`・`solvers`を省略または空配列にする |

SAのtemperaturesとEOのtausは必須。EO-SAはtaus・temperatures双方が必須であり、fitnesses（適応度）はEOと同じ。HCに温度・tauは指定できない。HC・SAのsmoothing省略時はnone一つ。Kを必要とする種類のみ非空のks配列を受け付ける。EO・EO-SAはfitnesses配列を受け付け、省略時は`{ kind = "default" }`一つへ解決する。指定する場合は非空とし、定義固有のparamsは各項目で明示する。組み込みdefaultのparamsは空のみ許可する。異なるparamsの比較は項目の列挙で行い、未知の定義は拒否する。

配列による明示列挙を唯一のスイープ構文とする。式評価は導入しない。グループごとに「グラフ×近傍方式×ソルバー設定×予算×探索シード」を展開し、全グループを連結する。EOはtaus×fitnesses、EO-SAはtaus×temperatures×fitnesses、HC・SAは手法パラメータ×smoothingを展開する。展開結果は条件ID・シード順に表示し、乱数と識別子は配列位置に依存させない。

`max_steps = [100, 1000]`は別条件の2予算へ展開する。グループの`budget`は全体の予算を置き換え、省略時は全体から継承する。全グループで上書きする場合も全体の`budget`は必須。グラフ・alpha・計測・探索シードは全グループで共有する。予算配列は昇順へ正規化し、要素1個ならスカラーへ正規化する。0、空配列、重複、およびグループをまたぐ同一の有効条件を拒否する。明示計測点はすべての有効予算以下でなければならず、自動で切り捨てない。

[rounds.toml](../examples/configs/rounds.toml)は、FlipのSAに10・30ステップ、SwapのEOに20・40ステップを割り当て、暫定解ベイスンも計測する24ジョブの実行例。`gpp run examples/configs/rounds.toml --rounds --deadline-seconds 60`でラウンド単位に実行できる。

次の例は1グラフ、2近傍方式、SA 6設定・HC 1設定・EO 2設定、探索3シードで合計54ジョブとなる。

```toml
schema_version = 1
name = "comparison"
run_seeds = [0, 1, 2]
neighborhoods = ["flip", "swap"]

[problem]
alpha = 0.05

[budget]
max_steps = 100

[measurement]
schedule = "logarithmic"
basin = "both"
max_basin_steps = 100
diagnostics = false

[[graphs]]
kind = "random"
node_counts = [12]
expected_degrees = [5.0]
seeds = [42]

[[solvers]]
kind = "sa"
temperatures = [0.1, 1.0]
[[solvers.smoothing]]
kind = "none"
[[solvers.smoothing]]
kind = "random_k_average"
ks = [8, 32]

[[solvers]]
kind = "hc"
[[solvers.smoothing]]
kind = "none"

[[solvers]]
kind = "eo"
taus = [1.2, 1.5]
[[solvers.fitnesses]]
kind = "default"
```

検証は展開後の全条件にも適用する。頂点数は2以上、次数は0以上n-1以下、各浮動小数点値と目的関数の上限は有限、シードと件数の算術はchecked演算とする。重複シード、同一グラフ、同一有効実行設定を拒否する。指数変換の旧設定は廃止し、温度とステップ数を実値で受け取る。実行中に非有限の評価値が発生した場合も、NaNの順位付けや保存を続けず、そのジョブを原因付きの数値エラーとして終了する。

## 5. 実行・計測・再現性

### 5.1 共通実行経路

CLIとライブラリは同じ設定検証、計画展開、単一実験、計測を呼び出す。ソルバーは探索状態を更新し、実験側が計測時点と停止を管理し、保存側が結果を確定する。

グラフごとに一つの読み取り専用Problemを共有し、探索状態・平滑化器・乱数・計測器はジョブが所有する。ジョブ間で可変RNGを共有しない。全近傍の解ベクトルを毎回作る経路を避け、整数状態と差分評価を利用する。HC・EOも同じ差分評価を利用する。

### 5.2 乱数

再現性の契約は「同じアルゴリズムバージョン・グラフ・条件・探索シードなら、並列数、実行順、表示名、計測頻度、計測有無にかかわらず同じ探索結果」である。実行時間とイベント到着順は一致対象から除く。

- 初期分割はグラフ内容・近傍方式・探索シードから導出した専用RNGで作り、同じ近傍方式では手法・パラメータを変えても同じ初期分割にする。Swapは半数ずつの所属を一様にシャッフルする。
- 探索の候補選択・受理、ランダム平滑化、探索の同点選択を別ストリームにする。導出元はグラフ内容、近傍方式、有効な手法・平滑化または適応度の条件とバージョン、探索シード、アルゴリズムバージョン。予算と計測条件は含めない。
- 現行解の計測RNGは探索から独立させ、各チェックポイントのステップ番号と用途から個別に導出する。他の計測点を追加しても共通の計測点の値が変わらないようにする。暫定解ベイスンはグラフ内容・近傍・alpha・探索シード・暫定分割と専用用途ラベル`basin-best-real-v1`から導出し、ステップ番号・予算・手法を含めない。
- 導出は固定のSHA-256と用途ラベルを使い、固定エンディアンで64bitシードへ変換する。ハッシュ・変換規則を仕様とテストで固定する。RNGは既存のMT19937-64を継続する。

完成版リファクタリングで導入した乱数設計は、それ以前の一部の結果を意図的に変えている。構造整理段階の旧80ケースの一致確認と、完成版の参照実装との一致確認を別の検証として扱う。その後の高速化および今回の3機能の統合では、既存の探索乱数設計を変更しない。

### 5.3 計測

初期状態0、対数刻みまたは明示されたステップ、終了時点を記録する。同じステップを重複保存しない。HCの早期終了後の値を、存在しない後続ステップへ埋めない。

各計測点には、ステップ終了時の現行解と、そのステップまでに実評価値が最小だった暫定解を保存する。これらは重複排除した`partitions`配列への`SolutionId`参照であり、同一の分割を複数回保存しない。暫定解は計測のないステップでも更新し、同点では最初に得た分割を維持する。初期点と終了点は必ず記録し、終了点の現行解・暫定解はトップレベルの参照と一致させる。各時点の実評価値、カット数、群サイズ、ペナルティは分割から読み込み時に算出する。

現行解は探索状態が保持する解だけを対象とし、SA・EO-SAで棄却した候補やベイスン計測中の解を暫定解へ混ぜない。計測間に改善後再び悪化した場合も、計測点の現行解の最小値で暫定解を代用しない。`SolutionId`の範囲、分割長、頂点順を読み込み時に検証する。

HC・SAの非自明な平滑化では現在の平滑化評価を保存し、ランダム平滑化だけは探索が保持する評価値も保存する。決定的平滑化では両者が同じなので複製しない。noneとweighted_averageの有効K=0は実評価から表示値を生成する。EO・EO-SAの平滑化関連列は非適用とする。

ランダム平滑化の計測では、チェックポイントごとに近傍番号の抽出集合を固定する。同じ番号集合を現在解・候補・ベイスン終点の評価に使う。Swapでは現在の各群を頂点番号順に並べた正規の近傍列に番号を対応させ、固定した評価関数として常に有効な交換を評価する。「探索が保持する確率的評価値」と「計測用の固定集合による現在評価」は別の測定値である。

ベイスン計測は探索と同じFlip/Swap近傍で行う。同点改善は計測専用RNGで一様選択し、探索へのフィードバックは行わない。実空間・平滑化空間からの終点を必要な空間で評価して保存する。最大走査数で打ち切り、local_optimumまたはstep_limitを保存し、打ち切り終点を局所最適解と表示しない。ベイスン終点の分割を保存しないため、測定したスコアは必要な一次データとして残す。

`best_basin=true`の場合は、各計測点の暫定分割から実目的関数によるベイスンを追加計測し、`basin_best`へ保存する。現行解の`basin=none`とも組み合わせられる。平滑化したソルバーでも暫定解ベイスンの平滑化評価は計算・保存しない。同じ暫定分割が続く間は直前の完了計測を再利用し、評価回数を増やさない。暫定分割が更新された後の最初の計測点で再計算する。測定値と終了理由、診断有効時だけ走査数を残す。終点の分割や群サイズ差は保存しない。計測中断時は未完成値を保存せず、キャッシュも更新しない。

noneとweighted_averageの有効K=0ではbothでも実空間を1回だけ計測し、対応する平滑化指標は読み込み時に補完する。EO・EO-SAのbothは実空間のみと解釈し、平滑化側は補完しない。有効な計測モードは条件から求め、別フィールドとして保存しない。非適用・未実施の値はJSONではキーを省略し、TSVでは空欄とする。NaNは保存しない。

measurement.diagnosticsは既定false。trueの場合だけ、最終累積値として適用移動数、探索・計測それぞれの目的関数実計算数、EO・EO-SAの頂点適応度実計算数、探索・計測の時間内訳を保存し、ベイスンごとの走査数も追加する。キャッシュ参照は実計算数に含めない。受理数・棄却数・受理率は既存値から生成する。診断の有無は探索の乱数と結果を変えない。通常保存する時間は単一実験全体のelapsed_msだけとし、グラフ生成・ディスク保存を除く。詳細な項目と省略規則は[出力仕様](output-format.md)に従う。

### 5.4 中断と再実行

Ctrl+Cの1回目で新規ジョブの開始を止め、実行中ジョブに協調的な中断を要求する。探索ステップだけでなく、近傍走査・平滑化の大量評価・ベイスン探索・グラフ生成のループも最大1,024要素ごとに確認する。ファイルの書き込み中は確定または失敗まで待つ。2回目のCtrl+Cは終了コード130で即時終了し、書きかけファイルは完了結果として扱わない。

中断時は可能なら最後の確定した現行解・暫定解と既存計測を最新の未完了マーカー内の部分結果として保存する。未開始ジョブの状態ファイルは作らず、条件とファイルの有無から判断する。最後の計測が未完了なら測定値を省略し、中断処理のために高コストなベイスン計測を新たに始めない。

`resume`は完全な結果と設定の一致を検証して再利用し、残りを先頭から実行する。中断地点からの継続ではないことをヘルプと進捗に表示する。最新の未完了試行だけを保持し、再試行時に置き換える。成功履歴や過去試行の全履歴は保存しない。

## 6. 保存・識別子・結果出力

保存JSON、識別子、原子的な確定、未完了マーカー、破損／未来版の扱い、TSV・metadataの列契約は[output-format.md](output-format.md)を正本とする。ここでは同じ仕様を再掲しない。アプリケーションは未展開の`ExperimentSpec`と固定versionsを一度だけ保存し、結果は`(condition_id, seed)`で解決する。派生スコアと集計表は保存せず、`RunView`とexportで生成する。`resume`は有効な完了結果を再利用し、未完了は先頭から再実行する。

## 7. リポジトリと内部アーキテクチャ

```text
gpp_utils/
├── AGENTS.md                      # 並列分担・モデル選択・検証方針・実験のベースライン
├── Cargo.toml / Cargo.lock
├── README.md
├── src/
│   ├── lib.rs / error.rs
│   ├── optimization/              # CancellationToken・用途別RNG
│   ├── graph_partition/           # graph.rs生成・state.rs差分更新
│   ├── smoothing/                 # 全平均・加重平均・距離1/2抽出
│   ├── fitness/                   # VertexFitness・FitnessFactory・登録
│   ├── solvers/                   # 共通EngineによるHC・SA・EO・EO-SA
│   ├── experiment/                # config・plan・runner・measurement・result
│   ├── storage/                   # グラフ・条件・バッチ・結果とatomic
│   ├── export/                    # TSV・列メタデータ
│   └── bin/gpp/main.rs            # 全サブコマンド
├── examples/
│   ├── configs/                   # minimal・54ジョブcomparison
│   ├── basic_usage.rs
│   ├── custom_fitness.rs          # Registryを共有する保存・再開例
│   └── bench_search.rs / bench_exact.rs
├── experiments_sa_eo/             # ベースラインと復旧runの仕様・生成・実行（RECOVERY_PLAN_v1.md）
├── tests/                         # CLI・実験受入・独立参照計算
├── docs/
│   ├── application-plan.md
│   ├── output-format.md
│   ├── algorithms.md
│   ├── extending.md / performance.md
│   ├── convert_v1.md              # 旧実験履歴の新形式への変換記録
│   └── verification.md
├── rust-toolchain.toml
├── scripts/check.py              # ローカルとCIの共通検証入口
└── .github/workflows/ci.yml
```

前案のようにソルバー本体と実験専用SAを別々に増やさず、移動・評価の高速なバックエンドを共通のソルバーから呼ぶ。汎用の全再計算経路は小規模テストの参照として使い、本番で同じアルゴリズムの制御ループを二重管理しない。

`PartitionState`は分割、総カット数、各群サイズ、頂点別カット数を所有し、外部から個別に書き換えられない。スコア計算と更新を同じ実装へ集約する。グラフは読み取り専用とし、解状態への変更でグラフを複製しない。

`Graph`の構造体フィールドは公開 API ではない。頂点数と正規化辺は`node_count()`と`edges()`で読み取り、任意の構造不変条件を破る直接変更はさせない。これはRust APIの破壊的変更だが、保存JSONのschema versionや既存の識別子を変える理由にはならない。

ライブラリの中心となる公開インターフェース:

- `ExperimentSpec`: 入力形式に対応する未検証の設定。
- `BudgetSweep`・`StepCounts`: 未展開のスカラーまたは配列予算。展開後の`Condition`はスカラーの`Budget`を持つ。
- `ConditionSweep`: 近傍・ソルバー・任意の予算を組み合わせる条件グループ。
- `RuntimeOptions`: 出力先、並列数、上書き・復旧、`rounds`と`round_deadline: Option<Duration>`。実験の科学的条件とは分離する。
- `StoredExperiment`: 未展開の正規化済み設定と固定バージョン。experiment.jsonへ保存する唯一の条件型。
- `compile_experiment(spec) -> Result<ExperimentPlan>`: 検証・正規化・メモリ上での展開。I/Oとグラフ生成を伴わず、展開済みPlan自体は保存しない。保存条件の再展開では固定バージョンを検証する。
- `run_one(graph, condition, seed, cancellation, registry) -> Result<RunResult>`: 保存先に依存しない単一実験。
- `run_batch(plan, runtime_options, cancellation, registry, event_sink) -> Result<BatchSummary>`: 保存・キャッシュ・並列実行を伴うアプリケーション処理。
- `RunResult`、`MeasurementRecord`、`RunTermination`: 一次データ中心の結果と終了状態。
- `RunView`: 条件・グラフ・結果から表示用の省略値と派生値を生成する読み取り層。`graph()`、`condition()`、`result()`、安全な`try_partition()`、`breakdown()`、`measurement()`、`records()`、`final_measurement()`を提供する。`partition()`は検証済みのSolutionId向けの利便 API で、不正なIDではpanicする。
- `ScoreBreakdown`: `real`、`cut_edges`、`size_a`、`size_b`、`balance_penalty`を型付きで返す派生値。`MeasurementView`は現行・暫定のbreakdown、平滑化／探索評価、三種類のベイスンを共有した型で返す。
- `BatchSummary`: メモリ上の戻り値と標準出力専用の集計。自動保存しない。
- `VertexFitness`・`FitnessFactory`・`FitnessRegistry`: 適応度の実装と登録。検証・実行・再開には同じ登録情報を渡し、未登録やバージョン不一致を拒否する。
- Storeの読み取りAPIと`export_tsv(...)`: CLI以外からも保存結果を利用する入口。

バッチのevent_sinkにはジョブの終了状態を渡し、探索状態やRNGへの可変参照は渡さない。単一実験の計測値はRunResultから取得する。I/Oや表示の例外で計算結果を書き換えない。ソルバー・平滑化はジョブ所有の可変オブジェクトとし、共有Mutexで内部RNGを保護する設計を廃止する。

CLIは引数と表示、experimentは実験の進行、solversは探索規則、graph_partitionは問題固有の計算、storage/exportはI/Oを担当する。exportの列順・型・metadataは単一の`ColumnSpec`定義から生成する。計算層からCLI・保存・TOMLパーサーへ依存しない。旧公開モジュール名への互換用再exportは設けない。

## 8. 開発と受入

エージェントの分担、モデル選択、統合責任は[AGENTS.md](../AGENTS.md)を正本とする。完了済みの段階的な実装・分担の履歴は[verification.md](verification.md#完成までの実装履歴2026-09-15)に置く。現在の変更では、計算、保存、schema、CLI、README、例、テストを同じ公開契約に更新し、`python scripts/check.py`を通す。

## 9. 完了条件と検証

### 計算の正しさ

- 小規模グラフで全分割を列挙し、実スコア・差分評価・キャッシュ更新を照合する。最適値は比較用に使い、確率的手法に毎回の最適解到達を要求しない。
- HC・SA・EO・EO-SAについて、同じ乱数ストリームを与えた参照実装と各ステップの状態・最良値を照合する。
- 全平滑化、Kの境界、距離2の重複除去、alpha=0、次数0・n-1、温度0、同点処理を検証する。
- 現在解と最良解の区別、受理・棄却・走査数、初期・最終計測、ベイスン打ち切りの意味を検証する。

- Swapの等分割維持、交換頂点同士に辺がある場合の差分評価、距離1/2近傍の個数と重複除去を検証する。
- EOのdefault・multiplicative・additive適応度・孤立頂点・順位抽選・Swapの条件付き抽選、配布CLIの組み込み登録が3種であること、テスト用独自定義の注入から再開・出力までを検証する。

### 再現性と実験機能

- スレッド数1と複数、設定の並べ替え、表示名変更で結果が変わらない。
- 計測なし・実空間のみ・両空間、計測点の追加・削除、診断の有効・無効で探索の最終解と最良解が変わらない。共通の計測点の値も一致する。
- ステップ上限を延長した実行は、元の上限まで同じ探索軌跡を持つ。
- TOMLとJSONの等価設定は同じ解決済み計画と識別子になる。例の展開件数が54になる。
- 重複、未知キー、非有限値、負値、桁あふれ、無効な手法パラメータを実験開始前に拒否する。

- taus欠落・空配列・負値または非有限値、EOへのsmoothing、削除したk_average、未登録適応度、奇数頂点のSwapを拒否する。
- eo_saのtaus・temperatures欠落・空配列・負値または非有限値、eo_saへのsmoothingを拒否する。

### 運用と出力

- 成功・スキップ・グラフ失敗・結果保存失敗・中断が混在するバッチの件数、通知、終了コードが一致する。
- Ctrl+Cが近傍走査とベイスン計測にも届き、部分結果が完了扱いされない。
- 再開で完了結果の再計算を避け、残りの再実行結果が通常実行と一致する。
- 同一ルートへの二重起動、プロセス異常終了、書きかけJSON、破損キャッシュ、置換失敗、結果確定後・未完了マーカー削除前の終了から適切に復旧する。
- エクスポートの列、行数、条件識別、省略フィールドの補完と空欄、打ち切り・中断表示、タブや改行を含む名前を検証する。
- 展開済み計画・状態一覧・集計表・条件コピー・終点スコアを正本に保存せず、初期・終了点と共有分割参照を正しく復元する。途中の最良値を計測点の最小値で代用しない。
- 診断無効を0と誤表示しない。TSVを削除・再生成して実験の再実行なしに同じ値を得られる。
- 全コマンドのhelp、JSON表示、エラー時の終了コード、README記載の操作を実行テストする。

### 性能と完成判定

- fmt、clippy（警告をエラー化）、全ターゲットのテスト、releaseビルドが成功する。
- 既存SAの同条件で計測なしの探索速度を同一環境・同一ビルド設定で比較する。ウォームアップ後3回の中央値を用い、10%超の性能低下が再現する場合は原因を解消する。
- 新仕様の計測は内容と乱数が変わるため、旧計測との単純な時間比を合否基準にしない。ベンチでは診断を有効にして探索と計測の時間を分け、代表的なHC・SA・EOと距離2平滑化の結果をベンチ記録に残す。
- GUI・SQA・未完成実験バイナリがビルド対象と公開APIに存在せず、全サンプルが実行可能である。
- 設定作成→検証→計画→並列実行→中断→再開→確認→TSV出力を、補助スクリプトや手作業のJSON修復なしに完了できる。

この一連の完了条件を満たした状態を、本計画における「完成したアプリケーション」とする。
