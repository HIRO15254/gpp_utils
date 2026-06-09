//! GPP 実験用 GUI（4 タブ構成）。
//!
//! - Graphs: プリセット N/D/方式/シードでグラフを生成・永続化し、選択する。
//! - Configs: SA 実行条件（Θ、10^N 反復、スムージング）の集合を編集する。
//! - Run: 選択中のグラフと対象 Config 群、シード範囲で一括実行する（裏スレッド）。
//! - Results: 完了済み結果を 6 トレースの log-log プロットおよび TSV で確認する。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use eframe::egui;
use egui::{Color32, CornerRadius, RichText, Stroke};
use egui_plot::{Line, LineStyle, Plot, PlotPoints};

use gpp_utils::batch::{BatchEvent, BatchSpec, run_batch};
use gpp_utils::file_utils::save_json;
use gpp_utils::graph_spec::{
    EXPECTED_DEGREES, GraphKind, GraphLibrary, GraphSpec, NODE_COUNTS, StoredGraph,
};
use gpp_utils::run_config::{
    ConfigSweep, RunConfig, SmoothingKind, SmoothingSpec, SolverKind, SolverSpec, DEFAULT_TAU,
};
use gpp_utils::run_executor::{RunResult, ResultStore, StepRecord};

const GNUPLOT_DIR: &str = "data/gnuplot";

const GRAPH_DIR: &str = "data/graphs";
const RESULT_DIR: &str = "data/results";
const TSV_DIR: &str = "data/tsv";

/// GUI で組んだ実行内容を CLI 用バッチ定義として書き出す既定パス。
const BATCH_FILE: &str = "data/batch.json";

const TRACE_NAMES: &[&str] = &[
    "current (smoothed)",
    "current (real)",
    "basin sm \u{2190} sm",
    "basin real \u{2190} sm",
    "basin sm \u{2190} real",
    "basin real \u{2190} real",
];

/// gnuplot のファイル名・タイトルに使う ASCII 版の trace 名。
const TRACE_FILE_NAMES: &[&str] = &[
    "current_smoothed",
    "current_real",
    "basin_sm_from_sm",
    "basin_real_from_sm",
    "basin_sm_from_real",
    "basin_real_from_real",
];

const TRACE_COLORS: &[Color32] = &[
    Color32::from_rgb(86, 156, 214),
    Color32::from_rgb(220, 100, 60),
    Color32::from_rgb(120, 200, 120),
    Color32::from_rgb(50, 140, 70),
    Color32::from_rgb(230, 180, 80),
    Color32::from_rgb(180, 110, 200),
];

/// Averaged ビュー用のコンフィグ色パレット（巡回利用）。
const CONFIG_COLORS: &[Color32] = &[
    Color32::from_rgb(86, 156, 214),
    Color32::from_rgb(220, 100, 60),
    Color32::from_rgb(120, 200, 120),
    Color32::from_rgb(230, 180, 80),
    Color32::from_rgb(180, 110, 200),
    Color32::from_rgb(100, 200, 200),
    Color32::from_rgb(220, 130, 90),
    Color32::from_rgb(200, 160, 200),
];

fn main() -> eframe::Result<()> {
    use std::sync::Arc;

    // wgpu のアダプタ選択をカスタマイズする。
    // eframe のデフォルトはハードウェア GPU を要求し、見つからないと
    // `NoSuitableAdapterFound` で起動に失敗する（GPU 非搭載の Azure VM 等）。
    // ここでは列挙された全アダプタからハードウェアを優先しつつ、無ければ
    // WARP（Windows 同梱のソフトウェアラスタライザ）にフォールバックする。
    let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.native_adapter_selector = Some(Arc::new(|adapters, _surface| {
            adapters
                .iter()
                .max_by_key(|a| match a.get_info().device_type {
                    eframe::wgpu::DeviceType::DiscreteGpu => 4,
                    eframe::wgpu::DeviceType::IntegratedGpu => 3,
                    eframe::wgpu::DeviceType::VirtualGpu => 2,
                    eframe::wgpu::DeviceType::Cpu => 1, // WARP ソフトウェアラスタライザ
                    eframe::wgpu::DeviceType::Other => 0,
                })
                .cloned()
                .ok_or_else(|| {
                    "利用可能な wgpu アダプタが見つかりません（ハードウェア・WARP とも不在）"
                        .to_owned()
                })
        }));
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 860.0])
            .with_title("GPP Experiment Runner"),
        // OpenGL 非対応環境でも動くよう wgpu (DX12/Vulkan/WARP) を使う。
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
        ..Default::default()
    };
    eframe::run_native(
        "GPP Experiment Runner",
        options,
        Box::new(|cc| {
            install_cjk_font(&cc.egui_ctx);
            let mut style = (*cc.egui_ctx.style()).clone();
            style.spacing.item_spacing = egui::vec2(6.0, 4.0);
            style.spacing.slider_width = 140.0;
            cc.egui_ctx.set_style(style);
            Ok(Box::new(App::new()))
        }),
    )
}

/// CJK / 記号グリフを含むシステムフォントを検出し、フォールバックとして登録する。
/// 候補が見つからなければ何もしない（egui のデフォルトのまま）。
fn install_cjk_font(ctx: &egui::Context) {
    const CANDIDATES: &[&str] = &[
        // Linux (Debian/Ubuntu)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
        "/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf",
        // macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        // Windows
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
    ];

    let (path, bytes) = match CANDIDATES
        .iter()
        .find_map(|p| std::fs::read(p).ok().map(|b| (*p, b)))
    {
        Some(v) => v,
        None => {
            eprintln!(
                "[gui] No CJK font found in known locations; falling back to default font."
            );
            return;
        }
    };

    let mut fonts = egui::FontDefinitions::default();
    fonts
        .font_data
        .insert("cjk_fallback".into(), egui::FontData::from_owned(bytes).into());
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("cjk_fallback".into());
    }
    ctx.set_fonts(fonts);
    eprintln!("[gui] Loaded CJK font: {}", path);
}

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Graphs,
    Configs,
    Run,
    Results,
}

/// Results タブの表示モード。
#[derive(PartialEq, Clone, Copy)]
enum ViewMode {
    /// 1 シードずつ独立した曲線として描画（旧来の動作）。
    Individual,
    /// 同一 Config に属する複数シードをステップごとに平均し、
    /// Config 単位の代表曲線として描画する。色 = Config、線種 = trace。
    AveragedByConfig,
}

#[derive(Default)]
struct RunStatus {
    in_progress: bool,
    total: usize,
    done: usize,
    skipped: usize,
    log: Vec<String>,
}

impl RunStatus {
    fn push_log(&mut self, msg: impl Into<String>) {
        self.log.push(msg.into());
        if self.log.len() > 200 {
            let drop = self.log.len() - 200;
            self.log.drain(0..drop);
        }
    }
}

struct App {
    library: GraphLibrary,
    store: ResultStore,

    // Graphs
    graphs: Vec<StoredGraph>,
    selected_graph: Option<usize>,
    /// Run タブで一括実行する対象として選んでいるグラフ（複数可）。
    graph_selected_for_run: Vec<bool>,
    new_kind: GraphKind,
    new_n_idx: usize,
    new_d_idx: usize,
    new_seed: u64,

    // Configs
    configs: Vec<RunConfig>,
    config_selected_for_run: Vec<bool>,
    next_config_id: usize,

    // Config sweep generator inputs (Configs タブ)
    sweep_solver: SolverKind,
    sweep_thetas: String,
    sweep_include_greedy: bool,
    sweep_iters: String,
    sweep_kind: SmoothingKind,
    sweep_ks: String,
    sweep_weights: String,
    /// EO sweep の τ 値リスト（カンマ区切り）。
    sweep_taus: String,

    // Run params
    start_seed: u64,
    num_seeds: usize,
    /// 並列ワーカ数（1 = 直列）。デフォルトは検出されたコア数。
    num_threads: usize,
    /// 検出されたコア数（UI スライダの上限に使う）。
    detected_cpus: usize,

    // Run status (shared with thread)
    run_status: Arc<Mutex<RunStatus>>,
    /// 実行キャンセル要求フラグ（裏スレッドの run_batch と共有）。
    cancel: Arc<AtomicBool>,

    // Results
    loaded_results: Vec<RunResult>,
    selected_result: Option<usize>,
    show_trace: [bool; 6],
    view_mode: ViewMode,
    /// Results タブで描画対象とするグラフの id（`GraphSpec::id()`）。
    /// `None` または該当データなしの場合は loaded_results 全体を見せる。
    results_graph_id: Option<String>,

    // UI
    active_tab: Tab,
    status: String,
}

/// カンマ／空白区切りの文字列を数値リストへパースする（パース不能な要素は無視）。
fn parse_num_list<T: std::str::FromStr>(s: &str) -> Vec<T> {
    s.split([',', ' ', '\t', '\n'])
        .filter_map(|tok| {
            let t = tok.trim();
            if t.is_empty() { None } else { t.parse::<T>().ok() }
        })
        .collect()
}

impl App {
    fn new() -> Self {
        let library = GraphLibrary::new(GRAPH_DIR);
        let _ = library.ensure_dir();
        let graphs = library.list();

        let store = ResultStore::new(RESULT_DIR);
        let _ = std::fs::create_dir_all(&store.base_dir);

        let mut configs = Vec::new();
        configs.push(RunConfig {
            name: "T=1, 10^4".into(),
            theta: Some(0.0),
            log10_iterations: 4,
            smoothing: SmoothingSpec::None,
            solver: SolverSpec::Sa,
        });
        configs.push(RunConfig {
            name: "T=0, 10^4 (greedy)".into(),
            theta: None,
            log10_iterations: 4,
            smoothing: SmoothingSpec::None,
            solver: SolverSpec::Sa,
        });
        let config_selected_for_run = vec![true; configs.len()];

        let detected_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);

        let mut graph_selected_for_run = vec![false; graphs.len()];
        if !graph_selected_for_run.is_empty() {
            graph_selected_for_run[0] = true;
        }

        let mut s = Self {
            library,
            store,
            selected_graph: if graphs.is_empty() { None } else { Some(0) },
            graph_selected_for_run,
            graphs,
            new_kind: GraphKind::Random,
            new_n_idx: 1,
            new_d_idx: 1,
            new_seed: 0,
            configs,
            config_selected_for_run,
            next_config_id: 3,
            sweep_solver: SolverKind::Sa,
            sweep_thetas: "-1, 0, 1".into(),
            sweep_include_greedy: false,
            sweep_iters: "4".into(),
            sweep_kind: SmoothingKind::None,
            sweep_ks: "4, 8".into(),
            sweep_weights: "0.25, 0.5, 1".into(),
            sweep_taus: "1.3, 1.4, 1.5".into(),
            start_seed: 0,
            num_seeds: 1,
            num_threads: detected_cpus,
            detected_cpus,
            run_status: Arc::new(Mutex::new(RunStatus::default())),
            cancel: Arc::new(AtomicBool::new(false)),
            loaded_results: Vec::new(),
            selected_result: None,
            show_trace: [true, true, false, true, false, true],
            view_mode: ViewMode::Individual,
            results_graph_id: None,
            active_tab: Tab::Graphs,
            status: "Ready.".into(),
        };
        s.refresh_graphs();
        s
    }

    fn refresh_graphs(&mut self) {
        self.graphs = self.library.list();
        // selected_graph (single, for "view") の整合性を維持。
        if let Some(i) = self.selected_graph {
            if i >= self.graphs.len() {
                self.selected_graph = if self.graphs.is_empty() { None } else { Some(0) };
            }
        } else if !self.graphs.is_empty() {
            self.selected_graph = Some(0);
        }
        // 実行用の選択 vec を長さ合わせ。新規グラフは未選択にする。
        if self.graph_selected_for_run.len() < self.graphs.len() {
            self.graph_selected_for_run.resize(self.graphs.len(), false);
        } else if self.graph_selected_for_run.len() > self.graphs.len() {
            self.graph_selected_for_run.truncate(self.graphs.len());
        }
        // 全部 false なら、現在「閲覧中」のグラフだけは選択状態にしておく。
        if !self.graph_selected_for_run.is_empty()
            && !self.graph_selected_for_run.iter().any(|x| *x)
        {
            if let Some(i) = self.selected_graph {
                if let Some(slot) = self.graph_selected_for_run.get_mut(i) {
                    *slot = true;
                }
            }
        }
    }

    fn ensure_config_selection_len(&mut self) {
        self.config_selected_for_run.resize(self.configs.len(), true);
    }

    fn current_graph(&self) -> Option<&StoredGraph> {
        self.selected_graph.and_then(|i| self.graphs.get(i))
    }

    fn generate_graph_clicked(&mut self) {
        let n = NODE_COUNTS[self.new_n_idx.min(NODE_COUNTS.len() - 1)];
        let d = EXPECTED_DEGREES[self.new_d_idx.min(EXPECTED_DEGREES.len() - 1)];
        let spec = GraphSpec {
            kind: self.new_kind,
            n,
            d,
            seed: self.new_seed,
        };
        match self.library.load_or_generate(spec) {
            Ok(_) => {
                self.refresh_graphs();
                if let Some(idx) = self.graphs.iter().position(|g| g.spec == spec) {
                    self.selected_graph = Some(idx);
                    if let Some(slot) = self.graph_selected_for_run.get_mut(idx) {
                        *slot = true;
                    }
                }
                self.status = format!("Graph ready: {}", spec.id());
            }
            Err(e) => self.status = format!("generate error: {}", e),
        }
    }

    fn start_run(&mut self) {
        let graphs: Vec<StoredGraph> = self
            .graphs
            .iter()
            .enumerate()
            .filter(|(i, _)| self.graph_selected_for_run.get(*i).copied().unwrap_or(false))
            .map(|(_, g)| g.clone())
            .collect();
        if graphs.is_empty() {
            self.status = "No graphs selected for run.".into();
            return;
        }
        let cfgs: Vec<RunConfig> = self
            .configs
            .iter()
            .enumerate()
            .filter(|(i, _)| self.config_selected_for_run.get(*i).copied().unwrap_or(false))
            .map(|(_, c)| c.clone())
            .collect();
        if cfgs.is_empty() {
            self.status = "No configs selected.".into();
            return;
        }
        if self.num_seeds == 0 {
            self.status = "num_seeds must be >= 1.".into();
            return;
        }

        let num_threads = self.num_threads.clamp(1, self.detected_cpus.max(1));
        let total = graphs.len() * cfgs.len() * self.num_seeds;
        {
            let mut s = self.run_status.lock().unwrap();
            if s.in_progress {
                self.status = "Already running.".into();
                return;
            }
            *s = RunStatus {
                in_progress: true,
                total,
                done: 0,
                skipped: 0,
                log: Vec::new(),
            };
        }

        // 選択中のグラフ・設定・シード範囲を共通バッチ定義へ変換する。
        let spec = BatchSpec {
            graphs: graphs.iter().map(|g| g.spec).collect(),
            configs: cfgs,
            config_sweep: None,
            seed_start: self.start_seed,
            seed_count: self.num_seeds,
        };
        let graph_dir = self.library.base_dir.clone();
        let store_dir = self.store.base_dir.clone();
        let status_arc = Arc::clone(&self.run_status);
        let cancel = Arc::clone(&self.cancel);
        cancel.store(false, Ordering::Relaxed);

        // 実行は CLI と共通の batch::run_batch に委ねる。駆動は 1 本の裏スレッドに
        // 任せ、UI スレッドはブロックしない。進捗イベントを run_status へ反映する。
        thread::spawn(move || {
            run_batch(
                &spec,
                &graph_dir,
                &store_dir,
                num_threads,
                true,
                cancel,
                |ev| {
                    let mut st = status_arc.lock().unwrap();
                    match ev {
                        BatchEvent::Started { total, graphs, configs, seeds, threads } => st
                            .push_log(format!(
                                "Starting {} runs ({} graphs x {} configs x {} seeds, {} threads)",
                                total, graphs, configs, seeds, threads
                            )),
                        BatchEvent::Skipped { graph, config, seed } => {
                            st.skipped += 1;
                            st.done += 1;
                            st.push_log(format!("skip {} / {} / seed={}", graph, config, seed));
                        }
                        BatchEvent::Done { graph, config, seed, elapsed_s, final_real } => {
                            st.done += 1;
                            st.push_log(format!(
                                "done {} / {} / seed={} ({:.1}s, real={:.2})",
                                graph, config, seed, elapsed_s, final_real
                            ));
                        }
                        BatchEvent::SaveError { message } => {
                            st.push_log(format!("save error: {}", message))
                        }
                        BatchEvent::PoolError { message } => {
                            st.push_log(format!("thread pool error: {}", message))
                        }
                        BatchEvent::GraphError { spec, message } => {
                            st.push_log(format!("graph error {}: {}", spec.id(), message))
                        }
                        BatchEvent::Finished => {
                            st.in_progress = false;
                            st.push_log("--- finished ---");
                        }
                    }
                },
            );
        });

        self.status = format!("Run started ({} threads).", num_threads);
    }

    fn cancel_run(&mut self) {
        let mut st = self.run_status.lock().unwrap();
        if st.in_progress {
            self.cancel.store(true, Ordering::Relaxed);
            st.push_log("cancel requested");
        }
    }

    /// 選択中のグラフ・設定・シード範囲を CLI 用バッチ定義 JSON として書き出す。
    /// 出力ファイルは `cli --batch <path>` でそのまま実行できる。
    fn export_batch_json(&mut self) {
        let graphs: Vec<GraphSpec> = self
            .graphs
            .iter()
            .enumerate()
            .filter(|(i, _)| self.graph_selected_for_run.get(*i).copied().unwrap_or(false))
            .map(|(_, g)| g.spec)
            .collect();
        if graphs.is_empty() {
            self.status = "No graphs selected for run.".into();
            return;
        }
        let configs: Vec<RunConfig> = self
            .configs
            .iter()
            .enumerate()
            .filter(|(i, _)| self.config_selected_for_run.get(*i).copied().unwrap_or(false))
            .map(|(_, c)| c.clone())
            .collect();
        if configs.is_empty() {
            self.status = "No configs selected.".into();
            return;
        }

        let spec = BatchSpec {
            graphs,
            configs,
            config_sweep: None,
            seed_start: self.start_seed,
            seed_count: self.num_seeds,
        };
        let path = Path::new(BATCH_FILE);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match save_json(&spec, path) {
            Ok(()) => self.status = format!("Batch spec exported: {}", BATCH_FILE),
            Err(e) => self.status = format!("export error: {}", e),
        }
    }

    /// Configs タブの sweep 入力をパースし、直積で生成した `RunConfig` 群を設定リストへ追記する。
    ///
    /// - SA: 温度 × 反復回数 × 平滑化。
    /// - EO: 反復回数 × τ（theta/smoothing は無視）。
    fn generate_sweep_configs(&mut self) {
        let log10_iterations = parse_num_list::<u32>(&self.sweep_iters);
        if log10_iterations.is_empty() {
            self.status = "Sweep: specify at least one log10(iter) value.".into();
            return;
        }

        let sweep = match self.sweep_solver {
            SolverKind::Sa => {
                let mut thetas: Vec<Option<f64>> = parse_num_list::<f64>(&self.sweep_thetas)
                    .into_iter()
                    .map(Some)
                    .collect();
                if self.sweep_include_greedy {
                    thetas.push(None);
                }
                let ks = parse_num_list::<usize>(&self.sweep_ks);
                let weights = parse_num_list::<f64>(&self.sweep_weights);

                if thetas.is_empty() {
                    self.status = "Sweep: specify at least one Theta (or enable greedy).".into();
                    return;
                }
                if self.sweep_kind.uses_k() && ks.is_empty() {
                    self.status = "Sweep: specify at least one K value for this smoothing.".into();
                    return;
                }
                if self.sweep_kind.uses_weight() && weights.is_empty() {
                    self.status =
                        "Sweep: specify at least one weight (0..1) for weighted smoothing.".into();
                    return;
                }
                ConfigSweep {
                    thetas,
                    log10_iterations,
                    smoothing_kind: self.sweep_kind,
                    ks,
                    weights,
                    solver_kind: SolverKind::Sa,
                    taus: vec![],
                }
            }
            SolverKind::SaSwap => {
                // スワップ近傍 SA: theta × iters（smoothing なし）。
                let mut thetas: Vec<Option<f64>> = parse_num_list::<f64>(&self.sweep_thetas)
                    .into_iter()
                    .map(Some)
                    .collect();
                if self.sweep_include_greedy {
                    thetas.push(None);
                }
                if thetas.is_empty() {
                    self.status = "Sweep: specify at least one Theta (or enable greedy).".into();
                    return;
                }
                ConfigSweep {
                    thetas,
                    log10_iterations,
                    smoothing_kind: SmoothingKind::None,
                    ks: vec![],
                    weights: vec![],
                    solver_kind: SolverKind::SaSwap,
                    taus: vec![],
                }
            }
            SolverKind::Eo | SolverKind::EoFlip => {
                let taus = parse_num_list::<f64>(&self.sweep_taus);
                if taus.is_empty() {
                    self.status = "Sweep: specify at least one tau value for EO.".into();
                    return;
                }
                ConfigSweep {
                    thetas: vec![],
                    log10_iterations,
                    smoothing_kind: SmoothingKind::None,
                    ks: vec![],
                    weights: vec![],
                    solver_kind: self.sweep_solver,
                    taus,
                }
            }
        };

        let generated = sweep.expand();
        let n = generated.len();
        self.configs.extend(generated);
        self.ensure_config_selection_len();
        self.status = format!("Generated {} configs from sweep.", n);
    }

    fn load_results_for_current(&mut self) {
        self.loaded_results.clear();
        self.selected_result = None;
        let graphs: Vec<StoredGraph> = self
            .graphs
            .iter()
            .enumerate()
            .filter(|(i, _)| self.graph_selected_for_run.get(*i).copied().unwrap_or(false))
            .map(|(_, g)| g.clone())
            .collect();
        if graphs.is_empty() {
            self.status = "No graphs selected for run (Run tab).".into();
            return;
        }
        let mut loaded = 0usize;
        let mut graphs_with_data: Vec<String> = Vec::new();
        for graph in &graphs {
            let mut graph_loaded = 0usize;
            for (i, cfg) in self.configs.iter().enumerate() {
                if !self.config_selected_for_run.get(i).copied().unwrap_or(false) {
                    continue;
                }
                for s_off in 0..self.num_seeds {
                    let seed = self.start_seed.wrapping_add(s_off as u64);
                    if let Some(r) = self.store.load(&graph.spec, cfg, seed) {
                        self.loaded_results.push(r);
                        loaded += 1;
                        graph_loaded += 1;
                    }
                }
            }
            if graph_loaded > 0 {
                graphs_with_data.push(graph.spec.id());
            }
        }
        // Results タブのグラフフィルタを正規化。
        // 既存値が今回のロード結果に含まれていなければ、最初に見つかったグラフへ。
        let cur_valid = self
            .results_graph_id
            .as_ref()
            .map(|id| graphs_with_data.iter().any(|g| g == id))
            .unwrap_or(false);
        if !cur_valid {
            self.results_graph_id = graphs_with_data.first().cloned();
        }
        if loaded > 0 {
            // selected_result は filtered ビュー上の最初の要素を指すように初期化。
            let active = self.results_graph_id.clone();
            self.selected_result = self
                .loaded_results
                .iter()
                .position(|r| match &active {
                    Some(id) => r.graph_spec.id() == *id,
                    None => true,
                });
            self.status = format!(
                "Loaded {} results from {} graph(s).",
                loaded,
                graphs_with_data.len()
            );
        } else {
            self.results_graph_id = None;
            self.status = "No matching results found (run first).".into();
        }
    }

    fn export_selected_tsv(&mut self) {
        let r = match self.selected_result.and_then(|i| self.loaded_results.get(i)) {
            Some(r) => r,
            None => {
                self.status = "Select a result first.".into();
                return;
            }
        };
        let path = PathBuf::from(TSV_DIR)
            .join(r.graph_spec.id())
            .join(r.config.id())
            .join(format!("seed_{}.tsv", r.seed));
        match self.store.export_tsv(r, &path) {
            Ok(_) => self.status = format!("TSV: {}", path.display()),
            Err(e) => self.status = format!("export error: {}", e),
        }
    }

    /// 現在の Results プロットを gnuplot で再現する。GUI と同じ系列・色・線種・
    /// 並び順で `<dir>/plot.gp` + 系列ごとの `*.dat` を生成し、gnuplot を起動
    /// して `plot.png` を作成する。GUI のビューモード（Individual / Avg）と
    /// **現在選択しているグラフ** に合わせて系列を切り替える。
    fn export_results_gnuplot(&mut self) {
        if self.loaded_results.is_empty() {
            self.status = "No results loaded.".into();
            return;
        }
        // フィルタが指す（= UI 上で見えている）グラフの id を出力先に使う。
        let graph_id = match self.results_graph_id.clone() {
            Some(id) => id,
            None => self.loaded_results[0].graph_spec.id(),
        };
        let view_tag = match self.view_mode {
            ViewMode::Individual => "individual",
            ViewMode::AveragedByConfig => "avg",
        };
        let dir = PathBuf::from(GNUPLOT_DIR)
            .join("results")
            .join(format!("{}__{}", graph_id, view_tag));

        // 描画する系列のメタデータ（GUI のプロットコードと一致するように構築）。
        let series = self.build_results_gnuplot_series();
        if series.is_empty() {
            self.status =
                "No visible series to export (check graph filter / trace toggles).".into();
            return;
        }

        if let Err(e) = write_results_gnuplot_files(&dir, &graph_id, &series) {
            self.status = format!("export error: {}", e);
            return;
        }

        let png_path = dir.join("plot.png");
        self.status = invoke_gnuplot(&dir, &png_path);
    }

    /// GUI のプロットと同じロジックで系列を構築する。Configs タブの並び順を
    /// そのまま反映し、現在のグラフフィルタにマッチするデータだけを採用する。
    fn build_results_gnuplot_series(&self) -> Vec<GnuplotSeries> {
        let active_id = self.results_graph_id.clone();
        let filtered: Vec<&RunResult> = self
            .loaded_results
            .iter()
            .filter(|r| match &active_id {
                Some(id) => r.graph_spec.id() == *id,
                None => true,
            })
            .collect();

        let mut series = Vec::new();
        match self.view_mode {
            ViewMode::Individual => {
                // selected_result がフィルタ範囲に含まれていればそれだけを、
                // そうでなければフィルタ全件を出力する（GUI と同じ規則）。
                let sel = self
                    .selected_result
                    .and_then(|i| self.loaded_results.get(i))
                    .filter(|r| match &active_id {
                        Some(id) => r.graph_spec.id() == *id,
                        None => true,
                    });
                let to_plot: Vec<&RunResult> = match sel {
                    Some(r) => vec![r],
                    None => filtered.clone(),
                };
                for r in to_plot {
                    for trace in 0..6 {
                        if !self.show_trace[trace] {
                            continue;
                        }
                        let pts: Vec<(usize, f64)> = r
                            .records
                            .iter()
                            .filter(|rec| rec.step >= 1)
                            .map(|rec| (rec.step, trace_value(rec, trace)))
                            .filter(|p| p.1.is_finite())
                            .collect();
                        if pts.is_empty() {
                            continue;
                        }
                        series.push(GnuplotSeries {
                            file_name: format!(
                                "{}__seed{}__{}.dat",
                                r.config.id(),
                                r.seed,
                                TRACE_FILE_NAMES[trace]
                            ),
                            title: format!(
                                "{} | s={} | {}",
                                r.config.name, r.seed, TRACE_FILE_NAMES[trace]
                            ),
                            color: TRACE_COLORS[trace],
                            dashtype: 1,
                            points: pts,
                        });
                    }
                }
            }
            ViewMode::AveragedByConfig => {
                let avgs = average_by_config(&filtered);
                for (cfg_idx, avg) in avgs.iter().enumerate() {
                    let color = CONFIG_COLORS[cfg_idx % CONFIG_COLORS.len()];
                    for trace in 0..6 {
                        if !self.show_trace[trace] {
                            continue;
                        }
                        let pts: Vec<(usize, f64)> = avg
                            .records
                            .iter()
                            .filter(|rec| rec.step >= 1)
                            .map(|rec| (rec.step, trace_value(rec, trace)))
                            .filter(|p| p.1.is_finite())
                            .collect();
                        if pts.is_empty() {
                            continue;
                        }
                        series.push(GnuplotSeries {
                            file_name: format!(
                                "{}__avg__{}.dat",
                                avg.config.id(),
                                TRACE_FILE_NAMES[trace]
                            ),
                            title: format!(
                                "{} (n={}) | {}",
                                avg.config.name, avg.seed_count, TRACE_FILE_NAMES[trace]
                            ),
                            color,
                            dashtype: trace_gnuplot_dt(trace),
                            points: pts,
                        });
                    }
                }
            }
        }
        series
    }

    /// 現在閲覧中のグラフを gnuplot スクリプト + データファイルとして出力し、
    /// gnuplot が PATH にあれば PNG を生成する。
    fn export_graph_gnuplot(&mut self) {
        let g = match self.current_graph() {
            Some(g) => g.clone(),
            None => {
                self.status = "No graph selected.".into();
                return;
            }
        };
        let dir = PathBuf::from(GNUPLOT_DIR).join(g.spec.id());
        match write_gnuplot_files(&g, &dir) {
            Ok(()) => {}
            Err(e) => {
                self.status = format!("gnuplot export error: {}", e);
                return;
            }
        }

        let png_name = format!("{}.png", g.spec.id());
        let png_path = dir.join(&png_name);

        // gnuplot を起動して PNG を生成する。
        self.status = invoke_gnuplot(&dir, &png_path);
    }
}

/// `<dir>/plot.gp` を gnuplot で実行し、結果に応じた状態メッセージを返す。
/// 成功時は生成された `png_path` を案内し、失敗時は手動実行の手順を含める。
fn invoke_gnuplot(dir: &Path, png_path: &Path) -> String {
    let invoke = std::process::Command::new("gnuplot")
        .arg("plot.gp")
        .current_dir(dir)
        .output();
    match invoke {
        Ok(out) if out.status.success() => format!("PNG: {}", png_path.display()),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            format!(
                "gnuplot failed (exit {}): {}. Script saved at {}/plot.gp",
                out.status,
                stderr.trim(),
                dir.display()
            )
        }
        Err(e) => format!(
            "Could not invoke gnuplot ({}). Script saved at {}/plot.gp; run `gnuplot plot.gp` from that directory.",
            e,
            dir.display()
        ),
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Repaint frequently while a run is in progress so progress is visible.
        let in_progress = self.run_status.lock().unwrap().in_progress;
        if in_progress {
            ctx.request_repaint_after(std::time::Duration::from_millis(150));
        }

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.heading("GPP Experiment Runner");
                ui.separator();
                ui.selectable_value(&mut self.active_tab, Tab::Graphs, "Graphs");
                ui.selectable_value(&mut self.active_tab, Tab::Configs, "Configs");
                ui.selectable_value(&mut self.active_tab, Tab::Run, "Run");
                ui.selectable_value(&mut self.active_tab, Tab::Results, "Results");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let st = self.run_status.lock().unwrap();
                    if st.in_progress {
                        ui.colored_label(
                            Color32::from_rgb(220, 180, 60),
                            format!("running {}/{}", st.done, st.total),
                        );
                    }
                });
            });
            ui.add_space(2.0);
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.add_space(2.0);
            ui.label(RichText::new(&self.status).italics().small());
            ui.add_space(2.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.active_tab {
            Tab::Graphs => self.tab_graphs(ui),
            Tab::Configs => self.tab_configs(ui),
            Tab::Run => self.tab_run(ui),
            Tab::Results => self.tab_results(ui),
        });
    }
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
impl App {
    fn tab_graphs(&mut self, ui: &mut egui::Ui) {
        ui.columns(2, |cols| {
            // Left: generation form + list
            let left = &mut cols[0];
            left.heading("Generate");
            egui::Grid::new("gen_grid").num_columns(2).spacing([8.0, 6.0]).show(left, |ui| {
                ui.label("Kind:");
                egui::ComboBox::from_id_salt("kind")
                    .selected_text(match self.new_kind {
                        GraphKind::Random => "Random (Erdos-Renyi)",
                        GraphKind::Geometric => "Geometric",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.new_kind, GraphKind::Random, "Random (Erdos-Renyi)");
                        ui.selectable_value(&mut self.new_kind, GraphKind::Geometric, "Geometric");
                    });
                ui.end_row();

                ui.label("N:");
                egui::ComboBox::from_id_salt("n")
                    .selected_text(format!("{}", NODE_COUNTS[self.new_n_idx]))
                    .show_ui(ui, |ui| {
                        for (i, n) in NODE_COUNTS.iter().enumerate() {
                            ui.selectable_value(&mut self.new_n_idx, i, format!("{}", n));
                        }
                    });
                ui.end_row();

                ui.label("D:");
                egui::ComboBox::from_id_salt("d")
                    .selected_text(format!("{}", EXPECTED_DEGREES[self.new_d_idx]))
                    .show_ui(ui, |ui| {
                        for (i, d) in EXPECTED_DEGREES.iter().enumerate() {
                            ui.selectable_value(&mut self.new_d_idx, i, format!("{}", d));
                        }
                    });
                ui.end_row();

                ui.label("Seed:");
                ui.add(egui::DragValue::new(&mut self.new_seed).speed(1));
                ui.end_row();
            });

            left.add_space(4.0);
            if left
                .add_sized(
                    [left.available_width(), 28.0],
                    egui::Button::new(RichText::new("Generate / Load").strong()),
                )
                .clicked()
            {
                self.generate_graph_clicked();
            }
            if left.button("Refresh list").clicked() {
                self.refresh_graphs();
            }

            left.add_space(8.0);
            left.separator();
            left.heading("Library");
            left.label(
                RichText::new(format!(
                    "{} graphs in {}.  Checkbox = include in Run; click name to view.",
                    self.graphs.len(),
                    GRAPH_DIR
                ))
                .small()
                .weak(),
            );
            // 「全選択 / 全解除」一括操作。
            left.horizontal(|ui| {
                if ui.small_button("Select all").clicked() {
                    for slot in &mut self.graph_selected_for_run {
                        *slot = true;
                    }
                }
                if ui.small_button("Clear").clicked() {
                    for slot in &mut self.graph_selected_for_run {
                        *slot = false;
                    }
                }
            });
            // 長さ整合（refresh_graphs を経由しない経路への保険）。
            if self.graph_selected_for_run.len() != self.graphs.len() {
                self.graph_selected_for_run.resize(self.graphs.len(), false);
            }
            egui::ScrollArea::vertical().id_salt("graphs_scroll").show(left, |ui| {
                for i in 0..self.graphs.len() {
                    let id = self.graphs[i].spec.id();
                    let edges = self.graphs[i].edge_count;
                    let n = self.graphs[i].spec.n;
                    let selected = self.selected_graph == Some(i);
                    let label = format!("{} ({} edges, n={})", id, edges, n);
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.graph_selected_for_run[i], "")
                            .on_hover_text("Include in Run");
                        if ui.selectable_label(selected, label).clicked() {
                            self.selected_graph = Some(i);
                        }
                    });
                }
            });

            // Right: visualization
            let right = &mut cols[1];
            right.heading("Selected Graph");
            let mut export_clicked = false;
            if let Some(g) = self.current_graph() {
                let info = format!(
                    "{}\n{} nodes, {} edges",
                    g.spec.id(),
                    g.spec.n,
                    g.edge_count
                );
                right.label(RichText::new(info).small().weak());
                right.horizontal(|ui| {
                    if ui
                        .button("Export PNG (gnuplot)")
                        .on_hover_text(
                            "Write nodes.dat / edges.dat / plot.gp under data/gnuplot/<id>/ and try to invoke gnuplot.",
                        )
                        .clicked()
                    {
                        export_clicked = true;
                    }
                });
                right.add_space(4.0);
                draw_graph(right, g);
            } else {
                right.colored_label(Color32::GRAY, "No graph selected.");
            }
            if export_clicked {
                self.export_graph_gnuplot();
            }
        });
    }

    fn tab_configs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Run Configurations");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("+ Add").clicked() {
                    let id = self.next_config_id;
                    self.next_config_id += 1;
                    self.configs.push(RunConfig {
                        name: format!("config #{}", id),
                        theta: Some(0.0),
                        log10_iterations: 4,
                        smoothing: SmoothingSpec::None,
                        solver: SolverSpec::Sa,
                    });
                    self.ensure_config_selection_len();
                }
            });
        });
        ui.label(
            RichText::new(
                "Theta = log10(T). Iterations = 10^N. T = 0 (greedy) when Theta is disabled.",
            )
            .small()
            .weak(),
        );
        ui.add_space(4.0);

        egui::CollapsingHeader::new("Generate from sweep (SA: Theta x iter x smoothing / EO: iter x tau)")
            .default_open(false)
            .show(ui, |ui| {
                ui.label(
                    RichText::new(
                        "Enter comma-separated values per axis; every combination is appended below.",
                    )
                    .small()
                    .weak(),
                );
                let sweep_is_eo =
                    matches!(self.sweep_solver, SolverKind::Eo | SolverKind::EoFlip);
                // smoothing 軸はプレーン SA（フリップ近傍）のみ。SaSwap は theta のみ。
                let sweep_show_smoothing = matches!(self.sweep_solver, SolverKind::Sa);
                egui::Grid::new("sweep_grid")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Solver:");
                        egui::ComboBox::from_id_salt("sweep_solver")
                            .selected_text(match self.sweep_solver {
                                SolverKind::Sa => "SA (flip)",
                                SolverKind::SaSwap => "SA swap",
                                SolverKind::Eo => "EO swap (tau)",
                                SolverKind::EoFlip => "EO flip (tau)",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.sweep_solver,
                                    SolverKind::Sa,
                                    "SA (flip)",
                                );
                                ui.selectable_value(
                                    &mut self.sweep_solver,
                                    SolverKind::SaSwap,
                                    "SA swap",
                                );
                                ui.selectable_value(
                                    &mut self.sweep_solver,
                                    SolverKind::Eo,
                                    "EO swap (tau)",
                                );
                                ui.selectable_value(
                                    &mut self.sweep_solver,
                                    SolverKind::EoFlip,
                                    "EO flip (tau)",
                                );
                            });
                        ui.end_row();

                        ui.label("log10(iter) values:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.sweep_iters)
                                .desired_width(240.0)
                                .hint_text("e.g. 4, 5"),
                        );
                        ui.end_row();

                        if sweep_is_eo {
                            // EO: τ 軸のみ（theta/smoothing は無視）。
                            ui.label("tau values:");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.sweep_taus)
                                    .desired_width(240.0)
                                    .hint_text("e.g. 1.3, 1.4, 1.5"),
                            );
                            ui.end_row();
                        } else {
                            ui.label("Theta values:");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.sweep_thetas)
                                    .desired_width(240.0)
                                    .hint_text("e.g. -1, 0, 1"),
                            );
                            ui.end_row();

                            ui.label("");
                            ui.checkbox(
                                &mut self.sweep_include_greedy,
                                "also include T = 0 (greedy)",
                            );
                            ui.end_row();

                            if sweep_show_smoothing {
                                ui.label("Smoothing kind:");
                                egui::ComboBox::from_id_salt("sweep_kind")
                                    .selected_text(self.sweep_kind.label())
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut self.sweep_kind,
                                            SmoothingKind::None,
                                            "none",
                                        );
                                        ui.selectable_value(
                                            &mut self.sweep_kind,
                                            SmoothingKind::KAverage,
                                            "kavg (det)",
                                        );
                                        ui.selectable_value(
                                            &mut self.sweep_kind,
                                            SmoothingKind::RandomKAverage,
                                            "rkavg (rand)",
                                        );
                                        ui.selectable_value(
                                            &mut self.sweep_kind,
                                            SmoothingKind::WeightedAverage,
                                            "wavg (weighted)",
                                        );
                                    });
                                ui.end_row();

                                if self.sweep_kind.uses_weight() {
                                    ui.label("Weight values (0..1):");
                                    ui.add(
                                        egui::TextEdit::singleline(&mut self.sweep_weights)
                                            .desired_width(240.0)
                                            .hint_text("e.g. 0.25, 0.5, 1"),
                                    );
                                } else {
                                    ui.label("K values:");
                                    ui.add_enabled(
                                        self.sweep_kind.uses_k(),
                                        egui::TextEdit::singleline(&mut self.sweep_ks)
                                            .desired_width(240.0)
                                            .hint_text("e.g. 4, 8, 16"),
                                    );
                                }
                                ui.end_row();
                            }
                        }
                    });
                if ui
                    .button(RichText::new("Generate configs").strong())
                    .on_hover_text(
                        "Append every combination (SA: Theta x iter x smoothing, EO: iter x tau) to the list below.",
                    )
                    .clicked()
                {
                    self.generate_sweep_configs();
                }
            });
        ui.add_space(6.0);

        self.ensure_config_selection_len();
        let mut remove: Option<usize> = None;
        let mut move_up: Option<usize> = None;
        let mut move_down: Option<usize> = None;
        let cfg_count = self.configs.len();

        egui::ScrollArea::vertical().id_salt("cfg_scroll").show(ui, |ui| {
            for (idx, cfg) in self.configs.iter_mut().enumerate() {
                egui::Frame::NONE
                    .fill(Color32::from_rgb(34, 38, 48))
                    .corner_radius(CornerRadius::same(6))
                    .stroke(Stroke::new(1.0, Color32::from_rgb(60, 64, 76)))
                    .inner_margin(8.0)
                    .outer_margin(egui::Margin::symmetric(0, 3))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            ui.add(egui::TextEdit::singleline(&mut cfg.name).desired_width(180.0));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.small_button("\u{2715}").on_hover_text("Remove").clicked() {
                                    remove = Some(idx);
                                }
                                ui.add_space(6.0);
                                let down_enabled = idx + 1 < cfg_count;
                                if ui
                                    .add_enabled(
                                        down_enabled,
                                        egui::Button::new("\u{25BC}").small(),
                                    )
                                    .on_hover_text("Move down")
                                    .clicked()
                                {
                                    move_down = Some(idx);
                                }
                                let up_enabled = idx > 0;
                                if ui
                                    .add_enabled(up_enabled, egui::Button::new("\u{25B2}").small())
                                    .on_hover_text("Move up")
                                    .clicked()
                                {
                                    move_up = Some(idx);
                                }
                                ui.label(RichText::new(cfg.id()).small().weak());
                            });
                        });

                        // Solver セレクタ。EO のときは τ スライダを表示し、
                        // theta / smoothing 行は無視される旨を示す。
                        ui.horizontal(|ui| {
                            ui.label("Solver:");
                            let sel_text = match cfg.solver {
                                SolverSpec::Sa => "SA (flip)".to_string(),
                                SolverSpec::SaSwap => "SA swap".to_string(),
                                SolverSpec::Eo { tau } => format!("EO swap (tau={:.2})", tau),
                                SolverSpec::EoFlip { tau } => format!("EO flip (tau={:.2})", tau),
                            };
                            egui::ComboBox::from_id_salt(format!("solver_{}", idx))
                                .selected_text(sel_text)
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(
                                            matches!(cfg.solver, SolverSpec::Sa),
                                            "SA (flip)",
                                        )
                                        .clicked()
                                    {
                                        cfg.solver = SolverSpec::Sa;
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(cfg.solver, SolverSpec::SaSwap),
                                            "SA swap (strict balance, Metropolis)",
                                        )
                                        .clicked()
                                    {
                                        cfg.solver = SolverSpec::SaSwap;
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(cfg.solver, SolverSpec::Eo { .. }),
                                            "EO swap (strict balance)",
                                        )
                                        .clicked()
                                        && !matches!(cfg.solver, SolverSpec::Eo { .. })
                                    {
                                        cfg.solver = SolverSpec::Eo { tau: DEFAULT_TAU };
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(cfg.solver, SolverSpec::EoFlip { .. }),
                                            "EO flip (penalty balance, SA-comparable basin)",
                                        )
                                        .clicked()
                                        && !matches!(cfg.solver, SolverSpec::EoFlip { .. })
                                    {
                                        cfg.solver = SolverSpec::EoFlip { tau: DEFAULT_TAU };
                                    }
                                });
                            if let SolverSpec::Eo { tau } | SolverSpec::EoFlip { tau } =
                                &mut cfg.solver
                            {
                                ui.add(egui::Slider::new(tau, 1.0..=2.0).text("tau"));
                            }
                        });

                        // theta は SA / SaSwap で有効（メトロポリス温度）。EO では無視。
                        if matches!(cfg.solver, SolverSpec::Sa | SolverSpec::SaSwap) {
                            ui.horizontal(|ui| {
                                let mut has_theta = cfg.theta.is_some();
                                if ui
                                    .checkbox(&mut has_theta, "use Theta")
                                    .on_hover_text(
                                        "If unchecked, T = 0 (no acceptance of worse moves)",
                                    )
                                    .changed()
                                {
                                    cfg.theta = if has_theta { Some(0.0) } else { None };
                                }
                                if let Some(t) = &mut cfg.theta {
                                    ui.add(
                                        // step_by は付けない（sweep 生成の細かい
                                        // theta 値が丸められて壊れるのを防ぐ）。
                                        egui::Slider::new(t, -3.0..=3.0)
                                            .text("Theta = log10(T)"),
                                    );
                                } else {
                                    ui.colored_label(Color32::GRAY, "T = 0");
                                }
                            });
                        } else {
                            ui.colored_label(
                                Color32::GRAY,
                                "Theta / Smoothing は EO では無視されます",
                            );
                        }

                        ui.horizontal(|ui| {
                            let mut n = cfg.log10_iterations as i32;
                            if ui
                                .add(egui::Slider::new(&mut n, 1..=8).text("log10(iter)"))
                                .changed()
                            {
                                cfg.log10_iterations = n.max(0) as u32;
                            }
                            ui.label(format!("= {} iterations", cfg.iterations()));
                        });

                        if matches!(cfg.solver, SolverSpec::Sa) {
                        ui.horizontal(|ui| {
                            ui.label("Smoothing:");
                            let label = match cfg.smoothing {
                                SmoothingSpec::None => "None".to_string(),
                                SmoothingSpec::KAverage(k) => format!("K-Avg (det) K={}", k),
                                SmoothingSpec::RandomKAverage(k) => format!("K-Avg (rand) K={}", k),
                                SmoothingSpec::WeightedAverage(w) => format!("Weighted w={:.2}", w),
                            };
                            egui::ComboBox::from_id_salt(format!("sm_{}", idx))
                                .selected_text(label)
                                .show_ui(ui, |ui| {
                                    let cur_k = match cfg.smoothing {
                                        SmoothingSpec::KAverage(k)
                                        | SmoothingSpec::RandomKAverage(k) => k,
                                        _ => 8,
                                    };
                                    let cur_w = match cfg.smoothing {
                                        SmoothingSpec::WeightedAverage(w) => w,
                                        _ => 0.5,
                                    };
                                    if ui
                                        .selectable_label(
                                            matches!(cfg.smoothing, SmoothingSpec::None),
                                            "None",
                                        )
                                        .clicked()
                                    {
                                        cfg.smoothing = SmoothingSpec::None;
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(cfg.smoothing, SmoothingSpec::KAverage(_)),
                                            "K-Avg (det)",
                                        )
                                        .clicked()
                                    {
                                        cfg.smoothing = SmoothingSpec::KAverage(cur_k);
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(cfg.smoothing, SmoothingSpec::RandomKAverage(_)),
                                            "K-Avg (rand)",
                                        )
                                        .clicked()
                                    {
                                        cfg.smoothing = SmoothingSpec::RandomKAverage(cur_k);
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(cfg.smoothing, SmoothingSpec::WeightedAverage(_)),
                                            "Weighted",
                                        )
                                        .clicked()
                                    {
                                        cfg.smoothing = SmoothingSpec::WeightedAverage(cur_w);
                                    }
                                });
                            match &mut cfg.smoothing {
                                SmoothingSpec::None => {}
                                SmoothingSpec::KAverage(k)
                                | SmoothingSpec::RandomKAverage(k) => {
                                    ui.add(egui::Slider::new(k, 1..=64).text("K"));
                                }
                                SmoothingSpec::WeightedAverage(w) => {
                                    // step_by は付けない: 設定すると egui が値を
                                    // その倍数へ丸めて書き戻し、sweep で生成した
                                    // 細かい重み（0.125 等）が壊れるため。
                                    ui.add(egui::Slider::new(w, 0.0..=1.0).text("weight"));
                                }
                            }
                        });
                        }
                    });
            }
        });

        if let Some(i) = remove {
            self.configs.remove(i);
            self.config_selected_for_run.remove(i);
        }
        // 並べ替え（remove と同フレームで起きてもクラッシュしないよう範囲チェック）。
        if let Some(i) = move_up {
            if i > 0 && i < self.configs.len() {
                self.configs.swap(i, i - 1);
                self.config_selected_for_run.swap(i, i - 1);
            }
        }
        if let Some(i) = move_down {
            if i + 1 < self.configs.len() {
                self.configs.swap(i, i + 1);
                self.config_selected_for_run.swap(i, i + 1);
            }
        }
    }

    fn tab_run(&mut self, ui: &mut egui::Ui) {
        self.ensure_config_selection_len();
        if self.graph_selected_for_run.len() != self.graphs.len() {
            self.graph_selected_for_run.resize(self.graphs.len(), false);
        }
        ui.columns(2, |cols| {
            let left = &mut cols[0];
            left.heading("Target graphs");
            left.label(
                RichText::new(
                    "Selected graphs are processed sequentially within the same run.",
                )
                .small()
                .weak(),
            );
            left.horizontal(|ui| {
                if ui.small_button("Select all").clicked() {
                    for slot in &mut self.graph_selected_for_run {
                        *slot = true;
                    }
                }
                if ui.small_button("Clear").clicked() {
                    for slot in &mut self.graph_selected_for_run {
                        *slot = false;
                    }
                }
            });
            egui::ScrollArea::vertical()
                .id_salt("run_graphs_scroll")
                .max_height(180.0)
                .show(left, |ui| {
                    if self.graphs.is_empty() {
                        ui.colored_label(Color32::GRAY, "No graphs in library.");
                        return;
                    }
                    for i in 0..self.graphs.len() {
                        let label = format!(
                            "{}  ({} edges, n={})",
                            self.graphs[i].spec.id(),
                            self.graphs[i].edge_count,
                            self.graphs[i].spec.n
                        );
                        ui.checkbox(&mut self.graph_selected_for_run[i], label);
                    }
                });
            let selected_graph_count =
                self.graph_selected_for_run.iter().filter(|x| **x).count();

            left.add_space(8.0);
            left.heading("Configs to run");
            egui::ScrollArea::vertical()
                .id_salt("run_configs_scroll")
                .max_height(180.0)
                .show(left, |ui| {
                    for i in 0..self.configs.len() {
                        let cfg_label =
                            format!("{}  ({})", self.configs[i].name, self.configs[i].id());
                        ui.checkbox(&mut self.config_selected_for_run[i], cfg_label);
                    }
                });
            let selected_config_count =
                self.config_selected_for_run.iter().filter(|x| **x).count();

            left.add_space(8.0);
            left.separator();
            left.heading("Seeds & threads");
            egui::Grid::new("seeds_grid").num_columns(2).spacing([8.0, 6.0]).show(left, |ui| {
                ui.label("Start seed:");
                ui.add(egui::DragValue::new(&mut self.start_seed).speed(1));
                ui.end_row();
                ui.label("# seeds:");
                ui.add(egui::Slider::new(&mut self.num_seeds, 1..=64));
                ui.end_row();
                ui.label("Threads:");
                let max_threads = self.detected_cpus.max(1);
                ui.add(
                    egui::Slider::new(&mut self.num_threads, 1..=max_threads)
                        .text(format!("of {}", max_threads)),
                );
                ui.end_row();
            });
            let total_jobs =
                selected_graph_count * selected_config_count * self.num_seeds.max(0);
            left.label(
                RichText::new(format!(
                    "Seeds: {}..{}, total jobs: {}",
                    self.start_seed,
                    self.start_seed.wrapping_add(self.num_seeds as u64),
                    total_jobs
                ))
                .small()
                .weak(),
            );

            left.add_space(8.0);
            let in_progress = self.run_status.lock().unwrap().in_progress;
            let can_run = !in_progress
                && selected_graph_count > 0
                && selected_config_count > 0
                && self.num_seeds > 0;
            left.horizontal(|ui| {
                if ui
                    .add_enabled(
                        can_run,
                        egui::Button::new(RichText::new("Run").strong())
                            .min_size(egui::vec2(100.0, 28.0)),
                    )
                    .clicked()
                {
                    self.start_run();
                }
                if ui
                    .add_enabled(in_progress, egui::Button::new("Cancel"))
                    .clicked()
                {
                    self.cancel_run();
                }
                let can_export =
                    selected_graph_count > 0 && selected_config_count > 0 && self.num_seeds > 0;
                if ui
                    .add_enabled(can_export, egui::Button::new("Export batch JSON"))
                    .on_hover_text(format!(
                        "Write the selected graphs/configs/seeds as a CLI batch file ({}).",
                        BATCH_FILE
                    ))
                    .clicked()
                {
                    self.export_batch_json();
                }
            });

            // Right: progress + log
            let right = &mut cols[1];
            right.heading("Progress");
            let (in_progress, total, done, skipped, log) = {
                let st = self.run_status.lock().unwrap();
                (st.in_progress, st.total, st.done, st.skipped, st.log.clone())
            };
            let pct = if total > 0 { done as f32 / total as f32 } else { 0.0 };
            right.add(egui::ProgressBar::new(pct).text(format!(
                "{}/{} (skipped {})",
                done, total, skipped
            )));
            right.add_space(4.0);
            right.label(
                RichText::new(if in_progress { "running..." } else { "idle" })
                    .small()
                    .weak(),
            );
            right.add_space(8.0);
            right.label("Log:");
            egui::ScrollArea::vertical()
                .id_salt("log_scroll")
                .stick_to_bottom(true)
                .max_height(right.available_height() - 20.0)
                .show(right, |ui| {
                    for line in log.iter() {
                        ui.label(RichText::new(line).monospace().small());
                    }
                });
        });
    }

    fn tab_results(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Results");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Export TSV").clicked() {
                    self.export_selected_tsv();
                }
                if ui
                    .button("Export gnuplot")
                    .on_hover_text(
                        "Reproduce the current plot with gnuplot: writes data files + plot.gp + tries to render plot.png.",
                    )
                    .clicked()
                {
                    self.export_results_gnuplot();
                }
                if ui.button("Load matching").clicked() {
                    self.load_results_for_current();
                }
            });
        });
        ui.label(
            RichText::new(
                "Loads runs in data/results matching the graphs/configs/seed range selected in the Run tab. Switch graphs with the Graph: selector below.",
            )
            .small()
            .weak(),
        );
        ui.add_space(4.0);

        // Results に存在するグラフ ID を初出順で集める（loaded_results の順は
        // load_results_for_current が「graphs × configs × seeds」で構築するので、
        // ここで初出順を取れば、グラフタブ／実行投入順に対応する。
        let loaded_graph_ids: Vec<String> = {
            let mut v: Vec<String> = Vec::new();
            for r in &self.loaded_results {
                let id = r.graph_spec.id();
                if !v.contains(&id) {
                    v.push(id);
                }
            }
            v
        };
        // フィルタが今のロード結果と整合しなければ修正する。
        let filter_valid = self
            .results_graph_id
            .as_ref()
            .map(|id| loaded_graph_ids.iter().any(|g| g == id))
            .unwrap_or(false);
        if !filter_valid {
            self.results_graph_id = loaded_graph_ids.first().cloned();
        }

        ui.horizontal(|ui| {
            ui.label("Graph:");
            let cur_label = self
                .results_graph_id
                .clone()
                .unwrap_or_else(|| "—".to_string());
            egui::ComboBox::from_id_salt("results_graph_filter")
                .width(280.0)
                .selected_text(cur_label)
                .show_ui(ui, |ui| {
                    if loaded_graph_ids.is_empty() {
                        ui.colored_label(Color32::GRAY, "(no results loaded)");
                    }
                    for gid in &loaded_graph_ids {
                        ui.selectable_value(
                            &mut self.results_graph_id,
                            Some(gid.clone()),
                            gid,
                        );
                    }
                });
            ui.label(
                RichText::new(format!("({} graph(s) loaded)", loaded_graph_ids.len()))
                    .small()
                    .weak(),
            );
        });

        ui.horizontal(|ui| {
            ui.label("View:");
            ui.selectable_value(&mut self.view_mode, ViewMode::Individual, "Individual");
            ui.selectable_value(
                &mut self.view_mode,
                ViewMode::AveragedByConfig,
                "Avg by config",
            );
            ui.separator();
            ui.label(
                RichText::new(match self.view_mode {
                    ViewMode::Individual => "Color = trace, one line per (result, trace).",
                    ViewMode::AveragedByConfig => {
                        "Color = config, line style = trace; mean over seeds per step."
                    }
                })
                .small()
                .weak(),
            );
        });

        ui.horizontal(|ui| {
            ui.label("Show traces:");
            for i in 0..6 {
                let mut on = self.show_trace[i];
                let resp = ui.checkbox(&mut on, "");
                if resp.changed() {
                    self.show_trace[i] = on;
                }
                ui.colored_label(TRACE_COLORS[i], TRACE_NAMES[i]);
            }
        });
        ui.add_space(4.0);

        let avail_h = ui.available_height();
        let plot_h = (avail_h * 0.55).max(220.0);

        // 現在のグラフフィルタにマッチする loaded_results のインデックス列。
        // インデックスは Copy なので、後で table 関数（&mut self）と共存可能。
        let active_id = self.results_graph_id.clone();
        let filtered_idx: Vec<usize> = self
            .loaded_results
            .iter()
            .enumerate()
            .filter(|(_, r)| match &active_id {
                Some(id) => r.graph_spec.id() == *id,
                None => true,
            })
            .map(|(i, _)| i)
            .collect();

        // Averaged ビュー用の集計はフィルタ後の結果に対して行う。
        // filtered_refs はこのブロックを抜けるとドロップされ、self の借用も解放。
        let averaged: Vec<AveragedSeries> = {
            let filtered_refs: Vec<&RunResult> = filtered_idx
                .iter()
                .map(|&i| &self.loaded_results[i])
                .collect();
            average_by_config(&filtered_refs)
        };

        Plot::new("results_plot")
            .height(plot_h)
            .x_axis_label("log\u{2081}\u{2080}(step)")
            .y_axis_label("score")
            .legend(egui_plot::Legend::default())
            .show(ui, |pui| match self.view_mode {
                ViewMode::Individual => {
                    // selected_result が現在のフィルタ範囲に含まれるなら 1 本だけ、
                    // そうでなければフィルタ全件を描画する。
                    let sel_in_filter = self
                        .selected_result
                        .filter(|i| filtered_idx.contains(i));
                    let indices: Vec<usize> = match sel_in_filter {
                        Some(i) => vec![i],
                        None => filtered_idx.clone(),
                    };
                    for i in indices {
                        let r = &self.loaded_results[i];
                        for trace in 0..6 {
                            if !self.show_trace[trace] {
                                continue;
                            }
                            let pts: Vec<[f64; 2]> = r
                                .records
                                .iter()
                                .filter(|rec| rec.step >= 1)
                                .map(|rec| {
                                    let y = trace_value(rec, trace);
                                    [(rec.step as f64).log10(), y]
                                })
                                .filter(|p| p[1].is_finite())
                                .collect();
                            if pts.is_empty() {
                                continue;
                            }
                            pui.line(
                                Line::new(PlotPoints::new(pts))
                                    .name(format!(
                                        "{} | {} | s={} | {}",
                                        r.config.name,
                                        r.config.id(),
                                        r.seed,
                                        TRACE_NAMES[trace]
                                    ))
                                    .color(TRACE_COLORS[trace])
                                    .width(1.5),
                            );
                        }
                    }
                }
                ViewMode::AveragedByConfig => {
                    for (cfg_idx, s) in averaged.iter().enumerate() {
                        let color = CONFIG_COLORS[cfg_idx % CONFIG_COLORS.len()];
                        for trace in 0..6 {
                            if !self.show_trace[trace] {
                                continue;
                            }
                            let pts: Vec<[f64; 2]> = s
                                .records
                                .iter()
                                .filter(|rec| rec.step >= 1)
                                .map(|rec| {
                                    let y = trace_value(rec, trace);
                                    [(rec.step as f64).log10(), y]
                                })
                                .filter(|p| p[1].is_finite())
                                .collect();
                            if pts.is_empty() {
                                continue;
                            }
                            pui.line(
                                Line::new(PlotPoints::new(pts))
                                    .name(format!(
                                        "{} (n={}) | {}",
                                        s.config.name, s.seed_count, TRACE_NAMES[trace]
                                    ))
                                    .color(color)
                                    .style(trace_line_style(trace))
                                    .width(1.8),
                            );
                        }
                    }
                }
            });

        ui.add_space(6.0);
        match self.view_mode {
            ViewMode::Individual => self.results_table_individual(ui, &filtered_idx),
            ViewMode::AveragedByConfig => self.results_table_averaged(ui, &averaged),
        }
    }

    fn results_table_individual(&mut self, ui: &mut egui::Ui, filtered_idx: &[usize]) {
        ui.heading("Loaded runs (current graph)");
        egui::ScrollArea::vertical().id_salt("results_scroll").show(ui, |ui| {
            if filtered_idx.is_empty() {
                ui.colored_label(Color32::GRAY, "No results loaded for this graph.");
                return;
            }
            egui::Grid::new("results_grid")
                .striped(true)
                .min_col_width(60.0)
                .show(ui, |ui| {
                    ui.label(RichText::new("sel").strong());
                    ui.label(RichText::new("config").strong());
                    ui.label(RichText::new("seed").strong());
                    ui.label(RichText::new("steps").strong());
                    ui.label(RichText::new("ms").strong());
                    ui.label(RichText::new("final real").strong());
                    ui.end_row();
                    for &i in filtered_idx {
                        let r = &self.loaded_results[i];
                        let is_sel = self.selected_result == Some(i);
                        let final_real = r
                            .records
                            .last()
                            .map(|x| x.current_real)
                            .unwrap_or(f64::NAN);
                        let cfg_label = format!("{} ({})", r.config.name, r.config.id());
                        let seed_str = format!("{}", r.seed);
                        let steps_str = format!("{}", r.records.len());
                        let ms_str = format!("{:.0}", r.elapsed_ms);
                        let final_str = format!("{:.2}", final_real);
                        if ui
                            .selectable_label(
                                is_sel,
                                if is_sel { "\u{25C9}" } else { "\u{25CB}" },
                            )
                            .clicked()
                        {
                            self.selected_result = if is_sel { None } else { Some(i) };
                        }
                        ui.label(cfg_label);
                        ui.label(seed_str);
                        ui.label(steps_str);
                        ui.label(ms_str);
                        ui.label(final_str);
                        ui.end_row();
                    }
                });
        });
    }

    fn results_table_averaged(&mut self, ui: &mut egui::Ui, averaged: &[AveragedSeries]) {
        ui.heading("Per-config averages");
        egui::ScrollArea::vertical().id_salt("avg_scroll").show(ui, |ui| {
            if averaged.is_empty() {
                ui.colored_label(Color32::GRAY, "No results loaded.");
                return;
            }
            egui::Grid::new("avg_grid")
                .striped(true)
                .min_col_width(60.0)
                .show(ui, |ui| {
                    ui.label(RichText::new("color").strong());
                    ui.label(RichText::new("config").strong());
                    ui.label(RichText::new("# seeds").strong());
                    ui.label(RichText::new("final real (mean)").strong());
                    ui.label(RichText::new("± std").strong());
                    ui.label(RichText::new("basin_real (mean)").strong());
                    ui.label(RichText::new("ms (mean)").strong());
                    ui.end_row();
                    for (idx, s) in averaged.iter().enumerate() {
                        let color = CONFIG_COLORS[idx % CONFIG_COLORS.len()];
                        ui.colored_label(color, "\u{25A0}");
                        ui.label(format!("{} ({})", s.config.name, s.config.id()));
                        ui.label(format!("{}", s.seed_count));
                        ui.label(format!("{:.3}", s.final_real_mean));
                        ui.label(format!("{:.3}", s.final_real_std));
                        ui.label(format!("{:.3}", s.final_basin_mean));
                        ui.label(format!("{:.0}", s.elapsed_ms_mean));
                        ui.end_row();
                    }
                });
        });
    }
}

fn trace_value(rec: &StepRecord, idx: usize) -> f64 {
    match idx {
        0 => rec.current_smoothed,
        1 => rec.current_real,
        2 => rec.basin_smoothed_from_smoothed,
        3 => rec.basin_real_from_smoothed,
        4 => rec.basin_smoothed_from_real,
        5 => rec.basin_real_from_real,
        _ => f64::NAN,
    }
}

/// Trace 番号ごとに線種を割り当てる（Averaged ビュー用）。
/// 線種は「smoothed 系 = 破線、real 系 = 実線、basin_real_from_smoothed 系
/// = 短い破線、basin_smoothed_from_real 系 = 点線」というルール。
fn trace_line_style(idx: usize) -> LineStyle {
    match idx {
        0 => LineStyle::Dashed { length: 8.0 },     // current (smoothed)
        1 => LineStyle::Solid,                       // current (real)
        2 => LineStyle::Dashed { length: 4.0 },     // basin sm <- sm
        3 => LineStyle::Solid,                       // basin real <- sm
        4 => LineStyle::Dotted { spacing: 6.0 },    // basin sm <- real
        5 => LineStyle::Solid,                       // basin real <- real
        _ => LineStyle::Solid,
    }
}

/// Config 単位の平均化結果。
struct AveragedSeries {
    config: RunConfig,
    seed_count: usize,
    /// 各ステップで全シードの算術平均をとった StepRecord。
    records: Vec<StepRecord>,
    final_real_mean: f64,
    final_real_std: f64,
    final_basin_mean: f64,
    elapsed_ms_mean: f64,
}

/// 同一 `config.id()` のシード結果群をステップ単位で平均化する。
///
/// 異なるシード間で `step` 集合が一致しない場合（途中キャンセル等）でも
/// step ごとに平均（観測されたシード数で除算）するため、欠損があっても
/// 落ちずに集計できる。
///
/// 出力順は `results` での **初出順** を維持する。`load_results_for_current`
/// が Configs タブの並び順で `loaded_results` を構築するため、結果として
/// 平均化系列も Configs タブと同じ並びになる。
///
/// 引数は参照のスライスなので、呼び出し側でグラフフィルタなどの絞り込みを
/// `Vec<&RunResult>` で前段集計してから渡せる（クローン不要）。
fn average_by_config(results: &[&RunResult]) -> Vec<AveragedSeries> {
    // 初出順を保持するために (id, RunConfig, Vec<&RunResult>) を Vec で持つ。
    let mut order: Vec<String> = Vec::new();
    let mut groups: BTreeMap<String, (RunConfig, Vec<&RunResult>)> = BTreeMap::new();
    for r in results {
        let id = r.config.id();
        if !groups.contains_key(&id) {
            order.push(id.clone());
            groups.insert(id.clone(), (r.config.clone(), Vec::new()));
        }
        groups.get_mut(&id).unwrap().1.push(*r);
    }

    let mut out = Vec::with_capacity(order.len());
    for id in order {
        let (cfg, group) = groups.remove(&id).expect("group present by construction");
        if group.is_empty() {
            continue;
        }

        // step → (sum of 6 fields, count)
        let mut by_step: BTreeMap<usize, ([f64; 6], usize)> = BTreeMap::new();
        for r in &group {
            for rec in &r.records {
                let entry = by_step.entry(rec.step).or_insert(([0.0; 6], 0));
                entry.0[0] += rec.current_smoothed;
                entry.0[1] += rec.current_real;
                entry.0[2] += rec.basin_smoothed_from_smoothed;
                entry.0[3] += rec.basin_real_from_smoothed;
                entry.0[4] += rec.basin_smoothed_from_real;
                entry.0[5] += rec.basin_real_from_real;
                entry.1 += 1;
            }
        }
        let records: Vec<StepRecord> = by_step
            .into_iter()
            .map(|(step, (sum, n))| {
                let nf = n as f64;
                StepRecord {
                    step,
                    current_smoothed: sum[0] / nf,
                    current_real: sum[1] / nf,
                    basin_smoothed_from_smoothed: sum[2] / nf,
                    basin_real_from_smoothed: sum[3] / nf,
                    basin_smoothed_from_real: sum[4] / nf,
                    basin_real_from_real: sum[5] / nf,
                }
            })
            .collect();

        // 最終ステップの実スコア統計（mean, std）と、平均経過時間。
        let finals_real: Vec<f64> = group
            .iter()
            .filter_map(|r| r.records.last().map(|x| x.current_real))
            .filter(|v| v.is_finite())
            .collect();
        let finals_basin: Vec<f64> = group
            .iter()
            .filter_map(|r| r.records.last().map(|x| x.basin_real_from_real))
            .filter(|v| v.is_finite())
            .collect();
        let elapsed: Vec<f64> = group.iter().map(|r| r.elapsed_ms).collect();

        let mean_or_nan = |v: &[f64]| -> f64 {
            if v.is_empty() {
                f64::NAN
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };
        let std_or_zero = |v: &[f64], mean: f64| -> f64 {
            if v.len() <= 1 {
                0.0
            } else {
                let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
                var.sqrt()
            }
        };

        let final_real_mean = mean_or_nan(&finals_real);
        let final_real_std = std_or_zero(&finals_real, final_real_mean);
        let final_basin_mean = mean_or_nan(&finals_basin);
        let elapsed_ms_mean = mean_or_nan(&elapsed);

        out.push(AveragedSeries {
            config: cfg,
            seed_count: group.len(),
            records,
            final_real_mean,
            final_real_std,
            final_basin_mean,
            elapsed_ms_mean,
        });
    }
    out
}

fn draw_graph(ui: &mut egui::Ui, g: &StoredGraph) {
    let coords = g.display_coords();
    let av = ui.available_size();
    let sz = av.x.min(av.y).max(120.0);
    let (resp, painter) =
        ui.allocate_painter(egui::vec2(sz, sz), egui::Sense::hover());
    let rect = resp.rect;
    painter.rect_filled(rect, CornerRadius::same(4), Color32::from_rgb(24, 26, 32));
    let inner = rect.shrink(10.0);
    let to_s = |x: f64, y: f64| {
        egui::pos2(
            inner.left() + x as f32 * inner.width(),
            inner.top() + (1.0 - y) as f32 * inner.height(),
        )
    };

    let n = g.spec.n;
    let ecol = Color32::from_rgba_premultiplied(100, 110, 130, 50);
    for u in 0..n {
        for &v in &g.adjacency_list[u] {
            if u < v {
                painter.line_segment(
                    [to_s(coords[u].0, coords[u].1), to_s(coords[v].0, coords[v].1)],
                    Stroke::new(0.6, ecol),
                );
            }
        }
    }
    let r = (3.0_f32).max(70.0 / (n as f32).sqrt());
    for i in 0..n {
        let p = to_s(coords[i].0, coords[i].1);
        painter.circle(
            p,
            r,
            Color32::from_rgb(160, 160, 170),
            Stroke::new(0.8, Color32::from_rgb(40, 42, 50)),
        );
    }
}

/// gnuplot 出力用の 1 系列（線）。
struct GnuplotSeries {
    /// 出力するデータファイル名（dir 配下、相対パス）。
    file_name: String,
    /// 凡例タイトル。
    title: String,
    /// 線色。
    color: Color32,
    /// gnuplot の `dt`（dashtype）番号。1 = solid。
    dashtype: u32,
    /// (step, score) のサンプル列。
    points: Vec<(usize, f64)>,
}

/// Trace 番号 → gnuplot dashtype。Averaged ビュー用。`trace_line_style` の
/// egui 側ルール（smoothed=破線, real=実線, basin_sm_from_real=点線）と
/// 視覚的に揃えてある。
fn trace_gnuplot_dt(idx: usize) -> u32 {
    match idx {
        0 => 2, // current (smoothed): dashed
        1 => 1, // current (real): solid
        2 => 5, // basin sm <- sm: long dash
        3 => 1, // basin real <- sm: solid
        4 => 3, // basin sm <- real: dotted
        5 => 1, // basin real <- real: solid
        _ => 1,
    }
}

/// Results プロットを再現する gnuplot スクリプトと、各系列のデータファイルを
/// `<dir>/` に書き出す。gnuplot の x 軸は `set logscale x 10` を使うので
/// データファイルには **生の step 値** をそのまま書く（GUI 側の
/// `(rec.step as f64).log10()` 変換は不要）。
fn write_results_gnuplot_files(
    dir: &Path,
    title_id: &str,
    series: &[GnuplotSeries],
) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;

    // 各系列のデータファイル。
    for s in series {
        let path = dir.join(&s.file_name);
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "# step\tscore")?;
        for (step, score) in &s.points {
            writeln!(f, "{}\t{}", step, score)?;
        }
    }

    // plot.gp 本体。
    let path = dir.join("plot.gp");
    let mut f = std::fs::File::create(&path)?;
    writeln!(f, "# Auto-generated by GPP Experiment Runner.")?;
    writeln!(f, "# Run from this directory: gnuplot plot.gp")?;
    writeln!(
        f,
        "set terminal pngcairo size 1200,800 enhanced background rgb 'white'"
    )?;
    writeln!(f, "set output 'plot.png'")?;
    writeln!(f, "set logscale x 10")?;
    writeln!(f, "set xlabel 'step (log scale)'")?;
    writeln!(f, "set ylabel 'score'")?;
    writeln!(f, "set grid")?;
    writeln!(f, "set key outside right top")?;
    writeln!(f, "set title '{}'", title_id.replace('\'', "''"))?;
    write!(f, "plot ")?;
    for (i, s) in series.iter().enumerate() {
        let [r, g, b, _] = s.color.to_array();
        let hex = format!("#{:02x}{:02x}{:02x}", r, g, b);
        let title = s.title.replace('\'', "''");
        write!(
            f,
            "'{}' using 1:2 with lines title '{}' lc rgb '{}' lw 1.6 dt {}",
            s.file_name, title, hex, s.dashtype
        )?;
        if i + 1 < series.len() {
            writeln!(f, ", \\")?;
            write!(f, "    ")?;
        } else {
            writeln!(f)?;
        }
    }
    Ok(())
}

/// `<dir>/{nodes.dat, edges.dat, plot.gp}` を生成する。座標は `display_coords()`
/// と同じ規則（[0,1]² 単位、Random は円配置、Geometric は格納座標）。
fn write_gnuplot_files(g: &StoredGraph, dir: &Path) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;
    let coords = g.display_coords();

    // nodes.dat: x\ty per node.
    let nodes_path = dir.join("nodes.dat");
    {
        let mut f = std::fs::File::create(&nodes_path)?;
        writeln!(f, "# x\ty")?;
        for (x, y) in &coords {
            writeln!(f, "{}\t{}", x, y)?;
        }
    }

    // edges.dat: x1\ty1\tdx\tdy per edge — gnuplot の `with vectors nohead` 用。
    let edges_path = dir.join("edges.dat");
    {
        let mut f = std::fs::File::create(&edges_path)?;
        writeln!(f, "# x1\ty1\tdx\tdy")?;
        for u in 0..g.spec.n {
            for &v in &g.adjacency_list[u] {
                if u < v {
                    let (x1, y1) = coords[u];
                    let (x2, y2) = coords[v];
                    writeln!(f, "{}\t{}\t{}\t{}", x1, y1, x2 - x1, y2 - y1)?;
                }
            }
        }
    }

    // plot.gp: gnuplot スクリプト本体。スクリプトディレクトリ内で実行する想定。
    let png_name = format!("{}.png", g.spec.id());
    let title = format!(
        "{}: {} nodes, {} edges",
        g.spec.id(),
        g.spec.n,
        g.edge_count
    );
    // ノード点サイズは N に応じて控えめにスケール。
    let pt_size: f64 = (1.5_f64).max(20.0 / (g.spec.n as f64).sqrt()).min(2.0);

    let script_path = dir.join("plot.gp");
    {
        let mut f = std::fs::File::create(&script_path)?;
        writeln!(f, "# Auto-generated by GPP Experiment Runner.")?;
        writeln!(f, "# Run from this directory: gnuplot plot.gp")?;
        writeln!(
            f,
            "set terminal pngcairo size 800,800 enhanced background rgb 'white'"
        )?;
        writeln!(f, "set output '{}'", png_name)?;
        writeln!(f, "set size square")?;
        writeln!(f, "set xrange [-0.05:1.05]")?;
        writeln!(f, "set yrange [-0.05:1.05]")?;
        writeln!(f, "unset xtics")?;
        writeln!(f, "unset ytics")?;
        writeln!(f, "unset border")?;
        writeln!(f, "unset key")?;
        writeln!(f, "set title '{}'", title.replace('\'', "''"))?;
        writeln!(
            f,
            "plot 'edges.dat' using 1:2:3:4 with vectors nohead lc rgb '#888888' lw 0.6, \\"
        )?;
        writeln!(
            f,
            "     'nodes.dat' using 1:2 with points pt 7 ps {:.2} lc rgb '#444444'",
            pt_size
        )?;
    }

    Ok(())
}
