# 拡張と互換性のチェックリスト

この文書は、標準実装に solver、fitness、計測、保存形式、TSV を追加または変更するときの正本である。計算規則は[algorithms.md](algorithms.md)、保存と列は[output-format.md](output-format.md)、通常の検証入口はリポジトリ直下の`python scripts/check.py`を使う。

## 変更前に決めること

1. solver または fitness の変更では、評価関数、乱数消費、同点処理、停止条件、適用できる近傍を定義する。`VertexFitness` は有限な頂点適応度を返し、小さい値が低適応度を意味する。登録名、意味バージョン、パラメータの正規化を決め、`FitnessRegistry` に登録する。
2. 計測の変更では、探索に影響しない専用 RNG、保存する一次データ、`RunView` が生成する派生値、キャンセル時に省略する値を決める。計測や診断の有無が探索軌跡を変えてはならない。
3. 保存形式の変更では、JSON schema version、ハッシュ／ID に含める条件、旧結果の読み取りまたは明示的な拒否を決める。`schema_version` を変える変更は、旧結果に対するユーザー操作を先に文書化する。
4. TSV の変更では、`ColumnSpec` を唯一の列定義として更新する。列順、型、単位、空欄規則、`metadata.json` を同時に更新し、手書きのヘッダーや説明を別に持たない。

## 変更から検証まで

| 変更 | version / key / schema | 既存結果への操作 | 必須検証 |
|---|---|---|---|
| 探索規則・候補順・浮動小数点の計算結果 | `versions.algorithm` | 旧条件と別ID。旧版対応を残さない場合は明示拒否 | 新しい独立参照、同点・孤立点・Flip/Swap |
| RNG本体・導出・消費規則 | `versions.rng`（探索規則も変われば`algorithm`も） | 旧系列を新系列として再利用しない | 各系列の状態、共通予算区間、計測・並列数への独立性 |
| 独自適応度の値・パラメータ解釈 | `versions["fitness:<登録名>"]`、Factoryの`version()` | 版不一致は再利用しない | 定義の独立参照、compile/run/resumeの同一Registry |
| 計測アルゴリズム・保存される計測値 | `versions.measurement` | 旧結果を新しい計測値として扱わない | 疎密な計測、診断有無、キャンセル、RunView |
| グラフ生成式・乱数消費 | `versions.generation` | 新しいgraph_id。旧グラフは無断再生成しない | 正規化辺、内容ハッシュ、生成の再現性 |
| 条件展開・正規化・既定値・ID符号化 | `versions.normalization` | 旧条件を旧規則で再展開するか、明示拒否 | TOML/JSON等価、並べ替え、重複、旧設定のID fixture |
| JSON一次データの互換性を壊す変更 | 対象の`schema_version`と、意味が変わる上記キー | 版ごとの読み取り実装または明示拒否。自動移行は現在未実装 | round trip、破損・未来版保全、resume/inspect |
| TSVの列・型・単位・意味の変更 | `metadata.json.schema_version`（出力版）と`ColumnSpec` | 一次結果は無効化せず、TSVを再生成 | 列順、空欄、全行、metadataの型・単位 |

キーの定義は`src/experiment/plan.rs::pinned_versions`にある。既存結果を変えない整理・高速化では版を上げず、旧IDとビット一致を確認する。意味を変える機能では旧参照を新実装に合わせて書き換えず、新バージョン用の独立参照を追加する。互換な省略可能フィールドの追加は、既定無効時のシリアライズと旧IDを維持できることをテストし、版を維持する判断をPRに明記する。意味が変わらない説明文の改善は出力版の変更を要しない。

## 実装箇所と受入条件

| 拡張 | 主な変更箇所 | 確認する連携 |
|---|---|---|
| ソルバー | `experiment/config.rs`のSweep/単一Spec、`plan.rs`の検証・展開、`solvers/engine.rs`の初期化・RNG・step | `runner.rs`の直接呼び出し検証、`result.rs`の適用条件、TSVの手法列、設定例 |
| 平滑化・近傍 | config/plan、`smoothing/mod.rs`、必要時`graph_partition/state.rs` | K正規化、候補列挙順、Flip/Swap不変条件、キャンセル周期、EO禁止条件 |
| 適応度 | `FitnessFactory`と`VertexFitness`を利用側で実装、Registry登録 | バージョン、params検証、全頂点の順序・有限性、再開時の登録一致 |
| 計測 | config、`runner.rs`の計測・専用RNG・キャッシュ、`result.rs`の型・検証・RunView | 初期/終了/中断、JSON省略、暫定解への非干渉、TSV、出力仕様 |
| 保存・再開 | `storage/mod.rs`の状態分類、`storage/atomic.rs`、`error.rs` | 完了結果優先、失敗記録、排他・原子的確定、破損と未対応版の区別 |
| TSV | `export/columns.rs`の表別の型付き列定義、`export/mod.rs`の値選択 | 順序・空欄・型・単位が同じ定義から出ること、RunViewでの補完 |

共通の計算型を増やすときは依存方向を[全体Plan](application-plan.md#7-リポジトリと内部アーキテクチャ)で先に決める。計算モジュールからCLIや保存処理を呼ばない。closed enumによる既存手法の列挙は維持し、拡張ごとに汎用プラグイン基盤を追加しない。

### 独自適応度の最小手順

実行・保存・再開までの例は[custom_fitness.rs](../examples/custom_fitness.rs)。`cargo run --locked --example custom_fitness`で一時ディレクトリに実行し、完了結果の再利用と未登録Registryの拒否を確認する。例の定義はCLIの組み込みに追加しない。

1. `FitnessFactory::validate`で未知キー、型、範囲を拒否する。検証は入力を正規化しないため、別表現を許す場合の同一性とIDを先に定義する。
2. `VertexFitness::values`は頂点番号順に頂点数ちょうどの有限値を返す。同じgraph/state/paramsには同じビット列を返し、外部の可変状態や探索RNGへ依存させない。計算結果を変えないキャッシュは許容する。
3. 独自登録名を使う。`register`は同名の旧Factoryを返して置換するAPIなので、重複を禁止する利用側は戻り値を確認する。`default`は固定された組み込み名として予約する。
4. `compile_experiment_with_versions(spec, &registry.versions())`の後、`storage::validate_registry`でFactoryのparamsも検証する。版の一覧だけではFactoryの検証処理は実行できない。
5. `run_one`/`run_batch`、保存後の`load_plan_with_registry`または`compile_stored_with_registry`へ同じ意味・版のRegistryを渡す。再開のために保存済みversion文字列だけを現行値へ書き換えてはならない。

### 変更別の追加検証

必須コマンドの正本は[scripts/check.py](../scripts/check.py)。Windows/LinuxのCIも同じ入口を使う。

- 計算変更: `src/solvers/exact_tests.rs`で全ステップ・RNG系列・f64ビット・診断を照合する。fixtureと共有する状態管理や平滑化を変える場合は、[performance.md](performance.md#検証と採用)に従い独立参照を追加する。
- 保存変更: `tests/storage_recovery.rs`と`tests/application.rs`で未来版、破損マーカー、完了結果保全、二重起動、中断を確認する。
- 出力変更: `tests/export_compatibility.rs`が凍結した旧exportと全TSVバイト・metadataの列名/型/単位を比較する。初期/終了、途中中断、診断無効、全平滑化・EOを含める。`tests/run_view.rs`と文書JSON例の検証も維持する。
- 拡張例: `examples/custom_fitness.rs`のテストとGraphのcompile-fail Rustdocを通常テストへ含める。新しい公開APIには前提・エラー・panic条件をRustdocへ記す。

保存状態の優先順位と復旧操作は[output-format.md](output-format.md#6-再開失敗を扱う最小の運用データ)を正本とする。未対応schemaは`error::UnsupportedSchema`という型で判定し、表示用エラー文言を処理の分岐条件に使わない。

## 公開 API とツールチェーン

公開型のフィールドを private にする、関数シグネチャを変える、または意味を変える変更は Rust API の破壊的変更である。利用者が必要とする getter、constructor、typed view を先に用意し、README の使用例、Rustdoc、`basic_usage` と `custom_fitness` を更新する。データ schema version と experiment / graph / condition ID は Rust API version と独立であり、Rust API だけの変更で保存済みデータを無効化しない。

公開モジュール内の`pub` APIは利用者向け契約として扱う。0.xでは破壊的変更時にCargoのminorを上げる。今回のGraphフィールド非公開化は0.2.0で行い、利用側は`graph.node_count`→`graph.node_count()`、`&graph.edges`→`graph.edges()`、辺の変更→`Graph::from_edges`での再構築へ移行する。`PartitionState`は構築に使った同じGraphで操作する。

`rust-toolchain.toml` の stable channel と Cargo の `rust-version = "1.97"` は対応ツールチェーンの契約である。対応範囲は current stable（少なくとも1.97）であり、過去 toolchain の独立保証はしない。toolchain または `Cargo.lock` を変える変更は、完全なチェックと固定参照を確認してレビューする。新しい toolchain をこの手順のためだけに自動インストールしない。

`Cargo.lock`はコミットし、依存更新は対象・理由を明示した変更として行う。通常検証は`--locked`とし、無関係な`cargo update`を行わない。ツールチェーンが変わる更新では実際の`rustc --version`を記録し、速度を比較する場合は新旧を同じ環境で測る。異なる環境間の浮動小数点完全一致を暗黙には保証しない。
