# 結果完全一致の高速化評価（2026-09-25）

基準ソース: `2fc65d9095d907bd600f75226ba904a9a349c934`。候補はこの
archive と `target/performance-20260925/baseline/` を起点にした。
全試作の統合版は`integrated-v1/`、採否を反映した最終Rustソースは`accepted-v1/`に凍結した。
20案を実装・測定し、下表の17案を全部または一部採用した。通常版の代表15条件は
速度比の中央値1.350倍、13条件で改善した。PGOはそこから全15条件で改善し、中央値1.194倍だった。

## 比較条件

`scripts/performance_cases.py`は`run_recovery_v1.py`のA・C・Bが参照する
生成済み仕様から条件を選ぶ。温度を転記・再計算せず、グラフシード0、
探索100万ステップ、実行シード0または1、56計測点、real/best basin、
max_basin_steps=10000を保つ。代表15条件と追加の18グラフ×4手法を用意する。
実験本体とは別に、これらのグラフ・保存分割で演算を反復するkernel測定を行う。
kernelの反復数は新しい科学的探索予算ではなく、実験結果として保存・混合しない。
保存先は`target/performance-20260925/`以下で、`data/v1`を変更しない。

比較では同じ `Cargo.lock`、Windows、Intel Core i7-10700KF（8 core/16 thread）、
Rust 1.97.0、release の fat LTO・codegen-units=1を使い、
ビルドと測定を重ねない。結果署名は時間と attempt ID だけを除外して照合する。
87 ケースの JSON subset は Rust 側でも元 TOML から絞った plan と condition ID、
graph ID、seed、全 f64 bit が一致することを検証した。独立参照の各ステップ・RNG、
TSV バイト、保存復旧契約を維持し、主受入は `python scripts/check.py` とする。

## 候補台帳

| ID | 案 | 採否・個別測定の根拠 |
|---|---|---|
| 01 | canonical近傍逐次列挙 | 採用。公開Vec APIを維持し、内部を逐次列挙するv2。対にした実行で高温Swap 1.195倍、全近傍平均 1.037倍。 |
| 02 | Swap隣接判定 bitset | 採用。判定kernelは0.976〜1.404倍。8 MiBを超える表は確保せず従来の疎表現へ戻す。 |
| 03 | EO同点 bucket の順序保持 bitset | 採用。個別比較の中央値1.316倍。`n <= 4096`のみ、32個以下のbucketと大規模入力はVecを使用。 |
| 04 | SA受理確率 | 採用。256枠のdirect mapでdeltaのbit列が一致した場合だけ元のexp結果を再利用。低温Flip 1.350倍、中温Flip 1.087倍。遅くなったHashMap試作は撤回。 |
| 05 | immutable Graph content hash | 採用。hash呼出kernelは540〜8249倍。グラフ全体の実行時間の倍率ではない。 |
| 06 | EO Swap 非空 bucket 索引 | 不採用。全8ケースで0.589〜0.963倍に悪化。 |
| 07 | EO実スコア二重計算除去 | 不採用。中央値0.996倍で明瞭な利益なし。 |
| 08 | 不変 CDF/LUT/計測点列 | 採用は計測点cacheのみ（呼出kernel約24倍）。CDF/LUT共有はab3で0.992〜1.019倍のため不採用。 |
| 09 | unique GraphCell 確保 | 採用。163,584ジョブのsetupで1.086倍。小バッチ全体では利益を確認できなかった。 |
| 10 | 検証済み結果の重複 validate 整理 | 採用: single-validation。除去した validate の単独コスト根拠はあるが、全体の有意効果は未確認。 |
| 11 | 分割 snapshot コピー・二重保持削減 | 不採用。測定中央値0.922倍で利益を確認できず、Vecと初出登録順の元構造へ復元。 |
| 12 | RunView 分割別 breakdown cache | 採用。同じ固定fixtureでの対比較は1.611〜4.007倍。公開constructorの検証を維持。 |
| 13 | TSV temporary file streaming | 採用: 速度はケースにより 0.938–1.144、メモリ保持量を優先。peak memory は before median 20.57 MB から 13.60 MB。 |
| 14 | TSV row 中間文字列削減 | 採用。速度の最大は1.07倍と限定的で、batch出力では0.959倍。peak pagefileは18.76 MB → streaming 13.03 MB → direct 12.15 MB。 |
| 15 | JSON 読込中間 DOM 削減 | 採用。RunResult限定のtyped読込は1.250〜2.117倍。不正・未来版・重複キーは元のValue経路に戻し、従来の分類とlast-winsを維持。 |
| 16 | smoothing basin clone 削減 | 採用: apply/undo。probe の向き補正後は 1.137–2.281。 |
| 17 | non-random smoothing indices | 採用。不要なindicesを作らない。個別比較の中央値1.022倍。 |
| 18 | basin 固定 sample plan | 採用: probe の向き補正後、K1 は 5.219、K32 は 3.302。 |
| 19 | 距離2 scratch / side / ordinal decoder | 採用。状態scratch再利用は中央値1.129倍、整数のpair逆引きは1.027倍。候補順と乱数消費は元と一致。 |
| 20 | PGO | 採用。通常の採用版に対して1.056〜1.694倍、中央値1.194倍。専用ビルド手順と復旧スクリプトの`--gpp`指定を追加。 |

## 統合後の測定

通常版はbefore/afterをABBA・BAABの順で各4回、PGO比較は各2回測定した。
ウォームアップ用実行は測定に含めない。表の通常版は`run_one`の探索・計測・
結果構築・最終検証を含み、グラフ生成とディスク保存は含まない。

| 代表条件 | 変更前 ms | 採用通常版 ms | 変更前÷通常版 | 通常版÷PGO版 |
|---|---:|---:|---:|---:|
| A 低温 SA Flip | 27.39 | 17.74 | 1.544 | 1.260 |
| A 高温 SA Swap | 7185.46 | 3696.60 | 1.944 | 1.365 |
| A 中温 SA Flip | 60.05 | 48.37 | 1.241 | 1.351 |
| A 中温 SA Swap | 400.10 | 306.16 | 1.307 | 1.385 |
| A EO Flip | 1400.85 | 804.85 | 1.741 | 1.175 |
| A EO Swap | 2836.32 | 1844.67 | 1.538 | 1.310 |
| A EO tau=0 | 852.32 | 902.02 | 0.945 | 1.056 |
| A EO tau=3 | 228.23 | 192.85 | 1.183 | 1.194 |
| B RandomK=1 | 250.72 | 185.76 | 1.350 | 1.151 |
| B RandomK=32 | 684.50 | 517.11 | 1.324 | 1.075 |
| B AllAverage | 2819.46 | 2303.19 | 1.224 | 1.694 |
| C multiply=0 | 2596.29 | 2721.81 | 0.954 | 1.066 |
| C multiply=0.5 | 600.30 | 399.11 | 1.504 | 1.286 |
| C add=0 | 284.68 | 172.49 | 1.650 | 1.168 |
| C add high | 1235.80 | 898.35 | 1.376 | 1.185 |

通常版では2条件が約5%遅くなる。追加72条件の一度ずつの走査でも、小さいグラフの
EO Swapに最大約12%の低下があった。この72条件の時間は交互比較ではなく、性能の
確定値として扱わない。低下を調べるため、bucket降格を32→16個にして表現切替を
減らす案（0.967〜1.119倍、8条件の大半はほぼ同じ）と、SA cacheをSAだけのBoxへ
移す案（5条件すべて0.972〜0.996倍）も実装・測定したが、いずれも追加採用しなかった。
入力ごとの全面的な高速化や、未測定の将来の条件での倍率は保証しない。

固定fixtureに対する統合後の読込は中央値1.480倍、RunViewは1.601倍、TSV出力は
1.073倍。汎用JSON読込は0.999倍、既存結果の再利用判定は0.966倍だった。

実`gpp run --rounds`も毎回新しい一時rootで比較した。SA低温の全32シードバッチは
変更前→通常版で1.444倍、通常版→PGO版で1.189倍。EO tau=0の8シードバッチは
通常版→PGO版で1.101倍。いずれも保存した全runの非時間フィールドが一致した。
これらは1 thread、各側2測定の壁時計時間であり、プロセス開始と保存を含む。

## PGO版の利用

PGOは[公式rustc手順](https://doc.rust-lang.org/rustc/profile-guided-optimization.html)に従い、
同じ凍結ソース・明示target・基本flagsから通常版、instrumented版、profile-use版を作る。
代表15条件で学習し、profileと各バイナリのSHA-256を残した。fast-mathは使わない。
学習していないCLI・batch専用関数のmissing-profile警告はあるが、比較に用いた
`bench_recovery`ではmissing-profile警告もhash mismatchもなかった。

この環境で作成したPGO版を、計算を開始せずに復旧計画と照合するコマンド:

```powershell
python experiments_sa_eo/run_recovery_v1.py --check --gpp `
  target/performance-20260925/pgo-v1/use-target/x86_64-pc-windows-msvc/release/gpp.exe
```

実験開始時は上の`--check`を外す。`--gpp`を省略した従来コマンドは通常releaseを
ビルドする。PGOはマシン・toolchain・ソースに依存するため、変更後は再生成して比較する。

```powershell
python scripts/performance_pgo.py --source path/to/frozen-source `
  --work target/pgo-new --cases target/performance-20260925/cases
```

`--work`には新規の空白を含まないパスを指定する。`llvm-tools-preview`が必要。
`metadata.json`に通常版・学習版・採用版の来歴を保存する。

## 受入確認

- 通常版の`python scripts/check.py`は全工程成功。
- PGO版も`accepted-v1/scripts/check.py`を、同じ明示target・profile-use flagsで全工程成功。
  fmt、locked clippy、全ターゲット・docテスト、release exact regression、release全ビルド、
  警告をエラーにするdoc生成を含む。
- 通常版とPGO版は代表15条件および追加72条件で、時間・attempt ID以外の結果署名が変更前と一致。
  PGOの追加72条件は学習対象外で、受入ビルドと並行して正しさだけを確認し、時間は評価から除外。
- 独立参照による各ステップ・f64 bit・乱数列、保存復旧、破損JSONの分類、TSV完全バイト一致の検査に成功。
- `run_recovery_v1.py --check`は既定版・PGO指定版とも成功。
  A 163,584、C 843,264、B 373,248ジョブと既存batch IDを維持。
- 計算・保存・PGO比較手順を実装担当以外がレビューし、結果互換性に関する指摘は解消済み。

科学仕様と本番データは変更せず、ベンチマークの成果物は`target/performance-20260925/`に分離した。

## 測定の読み方と限界

`speedup` は通常「before / after」の中央値である。basin probe だけは summary の
before が ON の main、after が OFF の ablation なので、採用候補の speedup はその
逆数として記した。batch の一部は command log に sidecar の入力 SHA がなく、queue の
同一 spec / fixture 引数で照合した限定証拠である。RunView の旧 baseline IO は fixture
SHA が残っておらず、ab2 比較を採否根拠にしていない。

単独測定は3または5反復、対比較はABBA/BAABで各4反復を使用した。Windowsの周波数変動、非 interleave の比較順、
fat LTO、I/O と OS cache の影響を完全には除けない。実験本体の 10^6 ステップを変える
測定はしていない。代表 15 条件に加え 18 graph × 4 手法の grid を用いるが、kernel は
診断用反復であり科学結果として保存・混合しない。

最初の ablation 群は再ビルド前の binary / provenance が混在したため無効化した。現在の
`results/*.jsonl` と `.commands.json` は再ビルド後の有効データだけを用い、比較時には
共通キー、spec SHA、fixture SHA（存在する場合）、安定署名を確認する。

## 再現と証跡

条件は `python scripts/performance_cases.py --out target/performance-20260925/cases --extended`
で作る。Rust 側の subset ID 検証は次で行う。

```powershell
cargo run --release --locked --example verify_performance_cases -- `
  --cases target/performance-20260925/cases --sources experiments_sa_eo
```

候補 build は `python scripts/performance_build.py build ...`、同一入力の paired 比較は
`python scripts/performance_compare.py paired ...`、通常の集約は
`python scripts/performance_compare.py summarize ...` を使う。export の保持量は
`python scripts/performance_memory.py --out ... -- path/to/bench_batch.exe ...` で記録する。
各buildのsource inputs SHA-256とoverlayは`bin/<tag>/build.json`、各比較binary、
case spec、fixture の SHA-256 は `results/*.commands.json` に残す。途中集約は
`results/ablation-first-half-summary.json` と
`results/ablation-second-half-summary.json`、memory の来歴はその JSON と memory 測定 JSON
を正本とする。最終要約の機械可読な証跡は
[performance-results-20260925.json](performance-results-20260925.json)に保存する。

## 2026-09-26の高速化（PR #9）

基準はEO-SAを加えたmaster `e4b6a1c`。作業中にmasterへ入った2026-09-25の高速化（`e4a7850`）を統合し、統合後は両方の結果とビット単位で一致させた。比較契約と実装方針は[performance.md](performance.md)に記す。環境はLinux（4コアの共有VM）、rustc 1.98.1、releaseのfat LTO・codegen-units=1。各担当の測定は他の作業と同時に行ったので、個別の倍率は目安である。命令数はcallgrind、時間は同じ入力で新旧を交互に実行した対の中央値。

### 候補台帳

| ID | 案 | 採否・個別測定の根拠 |
|---|---|---|
| S1 | 実評価の最良改善走査で、整数の下界から結果を変えられない候補を計算しない（ベイスンと平滑化なしHCで共有） | 採用。Swapのベイスン計測が約73〜115倍、代表110条件のCPU時間が177.9秒から45.1秒（他の1巡目の変更を含む）。 |
| S2 | 降下ごとに利得と利得水準のビット集合を保持し、移動ごとに更新 | 採用。n=500・次数10のSA 1ジョブの計測部分の命令数がFlipで5.3〜5.9分の1、Swapで3.5分の1。計測時間はFlip 2.5〜2.7倍、Swap 2.1〜2.2倍、HCの設定144件で2.1倍。 |
| S3 | 上限値を浮動小数点の推定から探し、正確な述語で確定 | 採用。Flipの1ジョブで上限計算の命令数が約770万から約250万。 |
| S4 | 走査のSIMD化、利得バケットによる記録点の直接探索、初回走査だけ追跡なし | 不採用。S2の後は走査がn/64語の読み取りになり、利益がない（初回だけ追跡なしは1回あたり約1.2万命令の増加）。 |
| E1 | EO索引をセルのビット集合と64ブロック単位の合計・非空マスクにする（Fenwick木を置き換え） | 採用。EO探索がFlip defaultで4.1倍、multiplicative・additiveで5.7〜5.8倍、Swapで3.1倍。 |
| E2 | EO索引の密な配置と、格納枠を使い回す配置の切り替え | 採用（メモリ）。n=4000・次数1000のmultiplicativeで索引が約600 MiBから21.5 MiB。基準の大きさ（n ≤ 500）では密な配置のまま。 |
| E3 | 状態を読まずに行を更新（Swapは触れた頂点を最終の行へ1回だけ移す）、グループ合計と非空フラグを分岐なしで更新、合計と格納枠の分離、`select`の全0語の読み飛ばし | 採用。n=500・次数10・τ1.5の1ステップあたり命令数がFlip defaultで12.9%、multiplicative・additiveで約14%減少。レビューの対比較（基準の大きさ96ケース、1スレッド、10^6ステップ）では1巡目の索引に対して幾何平均1.039倍、Flipで1.04〜1.08倍、τ ≥ 2.5の幾何グラフのSwapで1.03〜1.05倍、疎なランダムグラフのSwapは0.94〜0.98倍。Swapを2回のflipとして更新した最初の版は、共通の隣接頂点を往復させて高τの幾何グラフで7〜20%遅く、修正した。 |
| E4 | 格納枠を常に使う、3つの順位を別々に処理、Swapの適格ブロック表の縮小 | 不採用。命令数がそれぞれ15〜23%増、3279→3977、5071→5103。 |
| M1 | 平滑化で全近傍リストと添字配列を作らず、部分シャッフルをO(K)で再現、距離2を整数差分で計算 | 採用。平滑化つきSAがFlipで1.5〜3.3倍、Swapで6.6〜220倍。 |
| A1 | SAの手法判定を1回にし、`Engine::advance`で計測点までを専用ループで実行 | 採用。各変更を1つずつ外した比較で、Flip 1.09倍、Swap 1.08〜1.20倍。 |
| A2 | Metropolis判定の`exp`をdeltaのビット列で引くメモ（1024要素） | 採用。SA FlipがT=0.03で1.56倍、T=1で1.30倍。SwapとEO-SAはほぼ同等〜1.12倍。 |
| A3 | `apply_flip`の隣接頂点更新を分岐なしにする | 採用。SAのT=316でFlip 2.2倍・Swap 2.0倍、T=1のFlipで1.35倍、EO Swapで1.16倍。 |
| A4 | 隣接行列によるO(1)の隣接判定 | 採用。SA Swapで1.08〜1.15倍。上限は2026-09-25の作業に合わせて8 MiB（n ≤ 8192）。 |
| A5 | グラフ内容ハッシュのキャッシュ | 採用（2026-09-25の作業と同じ）。計測点ごとの再計算がなくなり、計測つきSA Flipで1.14倍。 |
| A6 | 専用ループを非インライン関数に分ける | 不採用。Flipで命令数が4.9%増。 |
| X1 | 平滑化ベイスンの候補を複製せず、適用・評価・取り消しで評価 | 採用（2026-09-25の作業の方式をM1の評価に合わせて移植）。 |
| X2 | Engineの分割状態を非公開にする | 採用（安全性）。S2とEO索引は手順だけが状態を変えることを前提にする。実行時の費用はない。 |

### 2026-09-25の作業との統合

同じ箇所を別方式で速くしていたので、速い方を残した。EOの`MemberSet`、SAの256枠のexpキャッシュ、平滑化の`MoveSequence`と固定サンプル計画はE1〜E3・A2・M1に置き換え、平滑化ベイスンの取り消し方式（X1）と隣接行列の上限は2026-09-25の作業に合わせた。置き換えた部分のテストは同じ意図のまま現行の実装に移し（expキャッシュの試験はメモへ、`MemberSet`の順序の試験は索引の再構築との照合へ）、凍結した平滑化モジュールとの比較テストは現行の平滑化に対して実行する。保存・出力・RunView・JSON読込・計測点キャッシュ・PGOの手順は変更していない。

### 統合後の測定

2026-09-27、Linux（4コアの共有VM、他の作業を止めた状態）、rustc 1.98.1、release。新master `e4a7850`と統合後の`528be21`を、同じ設定で交互に3回（平滑化ベイスンは1回）実行し、ジョブ時間の合計（`elapsed_ms`の和）の中央値を比べた。3並列で実行し、グラフ生成と保存は含まない。すべての設定で、時間と試行IDを除く結果JSONが一致した（918件）。

| 設定 | 内容 | `e4a7850` 秒 | `528be21` 秒 | 倍率 |
|---|---|---:|---:|---:|
| 代表条件 | SA 6温度・EO 5τ、Flip/Swap、Random n=124/250/500・次数10とGeometric n=500・次数5/20、10^6ステップ、56計測点、real/best basin（110件） | 128.93 | 18.59 | 6.94 |
| EO中心 | EO 4τ×5適応度、EO-SA、HC、SA、Random n=124/250・次数5/20とGeometric n=250、10^5ステップ、2シード（680件） | 111.97 | 18.62 | 6.01 |
| 段階C型 | EO Flip、default・multiplicative・additive、n=250/500（12件） | 16.64 | 3.02 | 5.51 |
| 段階B型 | SA、平滑化none・random_k（K=1/4/32）・all、n=250/500（20件） | 10.63 | 6.18 | 1.72 |
| 平滑化ベイスン | SA・HCと平滑化、basin both、n=64/124、2×10^4ステップ（96件、1回） | 604.58 | 91.63 | 6.60 |

代表条件の内訳は、探索が61.1秒から18.1秒（3.4倍）、計測が67.8秒から0.48秒（141倍）。統合後の時間の84%はEOの探索（Swap 58%、Flip 26%）、12%はSA Swapの探索で、計測は約3%になった。

### 打ち切りの判断

最後の測定と各担当・レビューの命令数の内訳から、この範囲での追加の高速化は見込めないと判断した。

- SAは1ステップの約半分が`gen_range`と受理判定の乱数で、乱数の引き方は比較契約で固定されている。
- EOのSwapの2頂点目は、非空ブロックを標準順序で走査する浮動小数点の順次和が約半分を占め、演算順は契約で固定されている。順位の更新は1頂点あたり数十命令で、残りの主な費用は配列の境界検査である。
- 計測の走査は、同点の乱数（仕様上必要）と降下データの更新が中心になった。

契約の外にある選択肢として、`unsafe`による境界検査の省略（EOの順位更新の10〜15%程度）、CPU固有の命令（popcnt等）を使うビルド、PGO（2026-09-25の手順）がある。いずれも保守方針・移植性・ビルド手順の判断を要するので、今回は行っていない。

### 受入確認

- `python scripts/check.py`がすべて成功した（Linux、rustc 1.98.1）。CIもWindows・Ubuntuで成功した。
- `python experiments_sa_eo/run_recovery_v1.py --check`で、ベースラインA（163,584ジョブ、`651e26af…`）と復旧C（`c27cf7e9…`）・B（`dec0166f…`）の`batch_id`とジョブ数が記録と一致した。
- 各変更を実装担当とは別のエージェントがレビューした。走査（1巡目・3巡目）、EO索引（1巡目・2巡目と修正）、平滑化、SA・状態の各レビューはいずれも承認で、推奨されたテストを取り込んだ。レビューでの照合は、1巡目の走査で`run_one` 71,134件と走査124万回、SA・状態で約2.27万件、3巡目の走査で3,045件と独立オラクルによる降下18,264回、EO索引の修正で7,908件で、すべて一致した。

