//! GPP 実験用 GUI（4 タブ構成）。
//!
//! - Graphs: プリセット N/D/方式/シードでグラフを生成・永続化し、複数選択する。
//! - Configs: 解法と平滑化のパラメータを複数値で指定し、全組合せの掃引を編集する。
//! - Run: 選択中のグラフ群と Sweep 群、シード範囲で一括並列実行する（rayon プール）。
//! - Results: 完了済み結果を 6 トレースの log-log プロットおよび TSV で確認する。

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use eframe::egui;
use egui::{Color32, CornerRadius, RichText, Stroke};
use egui_plot::{Line, Plot, PlotPoints};
use rayon::prelude::*;

use gpp_utils::graph_partition::GraphPartitionProblem;
use gpp_utils::graph_spec::{
    EXPECTED_DEGREES, GraphKind, GraphLibrary, GraphSpec, NODE_COUNTS, StoredGraph,
};
use gpp_utils::run_config::{ConfigSweep, RunConfig, SmoothingSpec};
use gpp_utils::run_executor::{ResultStore, RunResult, execute};

const GRAPH_DIR: &str = "data/graphs";
const RESULT_DIR: &str = "data/results";
const TSV_DIR: &str = "data/tsv";

const TRACE_NAMES: &[&str] = &[
    "current (smoothed)",
    "current (real)",
    "basin sm \u{2190} sm",
    "basin real \u{2190} sm",
    "basin sm \u{2190} real",
    "basin real \u{2190} real",
];

const TRACE_COLORS: &[Color32] = &[
    Color32::from_rgb(86, 156, 214),
    Color32::from_rgb(220, 100, 60),
    Color32::from_rgb(120, 200, 120),
    Color32::from_rgb(50, 140, 70),
    Color32::from_rgb(230, 180, 80),
    Color32::from_rgb(180, 110, 200),
];

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 860.0])
            .with_title("GPP Experiment Runner"),
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

#[derive(Default)]
struct RunStatus {
    in_progress: bool,
    total: usize,
    done: usize,
    skipped: usize,
    active_workers: usize,
    cancel: bool,
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

/// Configs タブ用の編集中バッファ。入力中の文字列はここに保持し、
/// `apply_to_sweep` で `ConfigSweep` に同期する。
#[derive(Clone)]
struct SweepInputs {
    thetas_text: String,
    iters_text: String,
    ks_text: String,
    /// [None, KAvg, RandomKAvg, Weighted] の順でチェック状態を保持。
    smoothing_kinds: [bool; 4],
    last_error: Option<String>,
}

impl SweepInputs {
    fn from_sweep(sw: &ConfigSweep) -> Self {
        let thetas_text = sw
            .thetas
            .iter()
            .map(|t| match t {
                None => "T0".to_string(),
                Some(v) => format_float(*v),
            })
            .collect::<Vec<_>>()
            .join(", ");
        let iters_text = sw
            .log10_iterations
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let mut kinds = [false; 4];
        let mut k_set = std::collections::BTreeSet::<usize>::new();
        for sm in &sw.smoothings {
            match sm {
                SmoothingSpec::None => kinds[0] = true,
                SmoothingSpec::KAverage(k) => {
                    kinds[1] = true;
                    k_set.insert(*k);
                }
                SmoothingSpec::RandomKAverage(k) => {
                    kinds[2] = true;
                    k_set.insert(*k);
                }
                SmoothingSpec::WeightedAverage(k) => {
                    kinds[3] = true;
                    k_set.insert(*k);
                }
            }
        }
        if k_set.is_empty() {
            k_set.insert(8);
        }
        let ks_text = k_set
            .iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        Self {
            thetas_text,
            iters_text,
            ks_text,
            smoothing_kinds: kinds,
            last_error: None,
        }
    }

    /// 入力テキストをパースして `ConfigSweep` に同期する。
    /// 何らかのエラーがあれば `last_error` にセットして false を返す。
    fn apply_to_sweep(&mut self, sw: &mut ConfigSweep) -> bool {
        let thetas = match parse_theta_list(&self.thetas_text) {
            Ok(v) => v,
            Err(e) => {
                self.last_error = Some(format!("Theta: {}", e));
                return false;
            }
        };
        let iters = match parse_u32_list(&self.iters_text) {
            Ok(v) => v,
            Err(e) => {
                self.last_error = Some(format!("log10(iter): {}", e));
                return false;
            }
        };
        let ks = match parse_usize_list(&self.ks_text) {
            Ok(v) => v,
            Err(e) => {
                self.last_error = Some(format!("K: {}", e));
                return false;
            }
        };

        let mut smoothings: Vec<SmoothingSpec> = Vec::new();
        if self.smoothing_kinds[0] {
            smoothings.push(SmoothingSpec::None);
        }
        let need_k = self.smoothing_kinds[1] || self.smoothing_kinds[2] || self.smoothing_kinds[3];
        if need_k && ks.is_empty() {
            self.last_error =
                Some("K values must be non-empty when a K-based smoothing is selected.".into());
            return false;
        }
        if self.smoothing_kinds[1] {
            for &k in &ks {
                smoothings.push(SmoothingSpec::KAverage(k));
            }
        }
        if self.smoothing_kinds[2] {
            for &k in &ks {
                smoothings.push(SmoothingSpec::RandomKAverage(k));
            }
        }
        if self.smoothing_kinds[3] {
            for &k in &ks {
                smoothings.push(SmoothingSpec::WeightedAverage(k));
            }
        }

        sw.thetas = thetas;
        sw.log10_iterations = iters;
        sw.smoothings = smoothings;
        self.last_error = None;
        true
    }
}

fn format_float(v: f64) -> String {
    if (v.fract()).abs() < 1e-9 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

fn parse_theta_list(s: &str) -> Result<Vec<Option<f64>>, String> {
    let mut out = Vec::new();
    for tok in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let lc = tok.to_ascii_lowercase();
        if matches!(lc.as_str(), "t0" | "off" | "none" | "-") {
            out.push(None);
        } else {
            let v = lc
                .parse::<f64>()
                .map_err(|_| format!("invalid number: {:?}", tok))?;
            out.push(Some(v));
        }
    }
    if out.is_empty() {
        return Err("at least one value required".into());
    }
    Ok(out)
}

fn parse_u32_list(s: &str) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for tok in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let v = tok
            .parse::<u32>()
            .map_err(|_| format!("invalid integer: {:?}", tok))?;
        out.push(v);
    }
    if out.is_empty() {
        return Err("at least one value required".into());
    }
    Ok(out)
}

fn parse_usize_list(s: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for tok in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let v = tok
            .parse::<usize>()
            .map_err(|_| format!("invalid integer: {:?}", tok))?;
        if v == 0 {
            return Err(format!("must be >= 1: {:?}", tok));
        }
        out.push(v);
    }
    Ok(out)
}

struct App {
    library: GraphLibrary,
    store: ResultStore,

    // Graphs
    graphs: Vec<StoredGraph>,
    selected_graph: Option<usize>,
    graph_selected_for_run: Vec<bool>,
    new_kind: GraphKind,
    new_n_idx: usize,
    new_d_idx: usize,
    new_seed: u64,

    // Configs (sweeps)
    sweeps: Vec<ConfigSweep>,
    sweep_inputs: Vec<SweepInputs>,
    sweep_selected_for_run: Vec<bool>,
    next_sweep_id: usize,

    // Run params
    start_seed: u64,
    num_seeds: usize,
    num_threads: usize,
    max_threads: usize,

    // Run status (shared with thread)
    run_status: Arc<Mutex<RunStatus>>,
    cancel_flag: Arc<AtomicBool>,

    // Results
    loaded_results: Vec<RunResult>,
    selected_result: Option<usize>,
    show_trace: [bool; 6],

    // UI
    active_tab: Tab,
    status: String,
}

impl App {
    fn new() -> Self {
        let library = GraphLibrary::new(GRAPH_DIR);
        let _ = library.ensure_dir();
        let graphs = library.list();

        let store = ResultStore::new(RESULT_DIR);
        let _ = std::fs::create_dir_all(&store.base_dir);

        let mut sweeps = Vec::new();
        sweeps.push(ConfigSweep {
            name: "default".into(),
            thetas: vec![Some(0.0)],
            log10_iterations: vec![4],
            smoothings: vec![SmoothingSpec::None],
        });
        sweeps.push(ConfigSweep {
            name: "greedy".into(),
            thetas: vec![None],
            log10_iterations: vec![4],
            smoothings: vec![SmoothingSpec::None],
        });
        let sweep_inputs = sweeps.iter().map(SweepInputs::from_sweep).collect();
        let sweep_selected_for_run = vec![true; sweeps.len()];

        let max_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);
        let num_threads = max_threads.min(4).max(1);

        let mut s = Self {
            library,
            store,
            selected_graph: if graphs.is_empty() { None } else { Some(0) },
            graph_selected_for_run: vec![true; graphs.len()],
            graphs,
            new_kind: GraphKind::Random,
            new_n_idx: 1,
            new_d_idx: 1,
            new_seed: 0,
            sweeps,
            sweep_inputs,
            sweep_selected_for_run,
            next_sweep_id: 3,
            start_seed: 0,
            num_seeds: 1,
            num_threads,
            max_threads,
            run_status: Arc::new(Mutex::new(RunStatus::default())),
            cancel_flag: Arc::new(AtomicBool::new(false)),
            loaded_results: Vec::new(),
            selected_result: None,
            show_trace: [true, true, false, true, false, true],
            active_tab: Tab::Graphs,
            status: "Ready.".into(),
        };
        s.refresh_graphs();
        s
    }

    fn refresh_graphs(&mut self) {
        let prev: Vec<(String, bool)> = self
            .graphs
            .iter()
            .zip(self.graph_selected_for_run.iter().copied())
            .map(|(g, sel)| (g.spec.id(), sel))
            .collect();
        self.graphs = self.library.list();
        // 既存の選択状態を id で引き継ぐ。新規はデフォルトで OFF。
        self.graph_selected_for_run = self
            .graphs
            .iter()
            .map(|g| {
                let id = g.spec.id();
                prev.iter()
                    .find(|(pid, _)| pid == &id)
                    .map(|(_, sel)| *sel)
                    .unwrap_or(false)
            })
            .collect();
        if let Some(i) = self.selected_graph {
            if i >= self.graphs.len() {
                self.selected_graph = if self.graphs.is_empty() { None } else { Some(0) };
            }
        } else if !self.graphs.is_empty() {
            self.selected_graph = Some(0);
        }
    }

    fn ensure_sweep_selection_len(&mut self) {
        self.sweep_selected_for_run.resize(self.sweeps.len(), true);
        while self.sweep_inputs.len() < self.sweeps.len() {
            let i = self.sweep_inputs.len();
            self.sweep_inputs.push(SweepInputs::from_sweep(&self.sweeps[i]));
        }
        self.sweep_inputs.truncate(self.sweeps.len());
    }

    fn current_graph(&self) -> Option<&StoredGraph> {
        self.selected_graph.and_then(|i| self.graphs.get(i))
    }

    fn selected_graphs_for_run(&self) -> Vec<StoredGraph> {
        self.graphs
            .iter()
            .enumerate()
            .filter(|(i, _)| self.graph_selected_for_run.get(*i).copied().unwrap_or(false))
            .map(|(_, g)| g.clone())
            .collect()
    }

    fn expanded_selected_configs(&self) -> Vec<RunConfig> {
        let mut out = Vec::new();
        for (i, sw) in self.sweeps.iter().enumerate() {
            if !self.sweep_selected_for_run.get(i).copied().unwrap_or(false) {
                continue;
            }
            out.extend(sw.expand());
        }
        // 同じ id を持つ重複を排除。
        let mut seen = std::collections::HashSet::new();
        out.retain(|c| seen.insert(c.id()));
        out
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
                    if let Some(flag) = self.graph_selected_for_run.get_mut(idx) {
                        *flag = true;
                    }
                }
                self.status = format!("Graph ready: {}", spec.id());
            }
            Err(e) => self.status = format!("generate error: {}", e),
        }
    }

    fn start_run(&mut self) {
        let graphs = self.selected_graphs_for_run();
        if graphs.is_empty() {
            self.status = "Check at least one graph in the Graphs tab.".into();
            return;
        }
        let cfgs = self.expanded_selected_configs();
        if cfgs.is_empty() {
            self.status = "No configs selected (or all sweeps are empty).".into();
            return;
        }
        if self.num_seeds == 0 {
            self.status = "num_seeds must be >= 1.".into();
            return;
        }
        let threads = self.num_threads.clamp(1, self.max_threads);

        // ワークアイテム: (graph_idx, cfg_idx, seed)
        let mut items: Vec<(usize, usize, u64)> = Vec::new();
        for gi in 0..graphs.len() {
            for ci in 0..cfgs.len() {
                for s_off in 0..self.num_seeds {
                    items.push((gi, ci, self.start_seed.wrapping_add(s_off as u64)));
                }
            }
        }
        let total = items.len();

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
                active_workers: 0,
                cancel: false,
                log: vec![format!(
                    "Starting {} runs on {} graphs × {} configs × {} seeds with {} threads",
                    total,
                    graphs.len(),
                    cfgs.len(),
                    self.num_seeds,
                    threads,
                )],
            };
        }
        self.cancel_flag.store(false, Ordering::SeqCst);

        // 共有データを Arc 化。
        let graphs_arc: Arc<Vec<StoredGraph>> = Arc::new(graphs);
        let problems_arc: Arc<Vec<GraphPartitionProblem>> =
            Arc::new(graphs_arc.iter().map(|g| g.problem()).collect());
        let cfgs_arc: Arc<Vec<RunConfig>> = Arc::new(cfgs);
        let items_arc: Arc<Vec<(usize, usize, u64)>> = Arc::new(items);
        let store_dir = self.store.base_dir.clone();
        let status_arc = Arc::clone(&self.run_status);
        let cancel_flag = Arc::clone(&self.cancel_flag);
        let active_counter = Arc::new(AtomicUsize::new(0));

        thread::spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build();
            let pool = match pool {
                Ok(p) => p,
                Err(e) => {
                    let mut st = status_arc.lock().unwrap();
                    st.in_progress = false;
                    st.push_log(format!("thread pool error: {}", e));
                    return;
                }
            };

            pool.install(|| {
                let store = ResultStore::new(&store_dir);
                items_arc.par_iter().for_each(|&(gi, ci, seed)| {
                    if cancel_flag.load(Ordering::SeqCst) {
                        let mut st = status_arc.lock().unwrap();
                        st.done += 1;
                        return;
                    }
                    let graph = &graphs_arc[gi];
                    let cfg = &cfgs_arc[ci];
                    let problem = &problems_arc[gi];

                    if store.exists(&graph.spec, cfg, seed) {
                        let mut st = status_arc.lock().unwrap();
                        st.skipped += 1;
                        st.done += 1;
                        st.push_log(format!(
                            "skip {} / {} / seed={}",
                            graph.spec.id(),
                            cfg.id(),
                            seed
                        ));
                        return;
                    }

                    active_counter.fetch_add(1, Ordering::SeqCst);
                    {
                        let mut st = status_arc.lock().unwrap();
                        st.active_workers = active_counter.load(Ordering::SeqCst);
                    }

                    let t0 = std::time::Instant::now();
                    let result = execute(graph.spec, cfg, problem, seed);
                    let elapsed = t0.elapsed().as_secs_f64();
                    let save_err = store.save(&result).err();

                    active_counter.fetch_sub(1, Ordering::SeqCst);
                    let mut st = status_arc.lock().unwrap();
                    st.active_workers = active_counter.load(Ordering::SeqCst);
                    st.done += 1;
                    if let Some(e) = save_err {
                        st.push_log(format!("save error: {}", e));
                    }
                    st.push_log(format!(
                        "done {} / {} / seed={} ({:.1}s, final real={:.2})",
                        graph.spec.id(),
                        cfg.id(),
                        seed,
                        elapsed,
                        result
                            .records
                            .last()
                            .map(|r| r.current_real)
                            .unwrap_or(f64::NAN)
                    ));
                });
            });

            let mut st = status_arc.lock().unwrap();
            st.in_progress = false;
            st.active_workers = 0;
            if cancel_flag.load(Ordering::SeqCst) {
                st.push_log("--- cancelled ---");
            } else {
                st.push_log("--- finished ---");
            }
        });

        self.status = format!("Run started ({} tasks, {} threads).", total, threads);
    }

    fn cancel_run(&mut self) {
        let in_progress = self.run_status.lock().unwrap().in_progress;
        if in_progress {
            self.cancel_flag.store(true, Ordering::SeqCst);
            let mut st = self.run_status.lock().unwrap();
            st.cancel = true;
            st.push_log("cancel requested");
        }
    }

    fn load_results_for_current(&mut self) {
        self.loaded_results.clear();
        self.selected_result = None;
        let graphs = self.selected_graphs_for_run();
        if graphs.is_empty() {
            self.status = "Check at least one graph in the Graphs tab.".into();
            return;
        }
        let cfgs = self.expanded_selected_configs();
        if cfgs.is_empty() {
            self.status = "No configs selected.".into();
            return;
        }
        let mut loaded = 0usize;
        for graph in &graphs {
            for cfg in &cfgs {
                for s_off in 0..self.num_seeds {
                    let seed = self.start_seed.wrapping_add(s_off as u64);
                    if let Some(r) = self.store.load(&graph.spec, cfg, seed) {
                        self.loaded_results.push(r);
                        loaded += 1;
                    }
                }
            }
        }
        if loaded > 0 {
            self.selected_result = Some(0);
            self.status = format!("Loaded {} results.", loaded);
        } else {
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
                            format!(
                                "running {}/{} ({} workers)",
                                st.done, st.total, st.active_workers
                            ),
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
            left.horizontal(|ui| {
                if ui.button("Refresh list").clicked() {
                    self.refresh_graphs();
                }
                if ui.button("Check all").clicked() {
                    for v in self.graph_selected_for_run.iter_mut() {
                        *v = true;
                    }
                }
                if ui.button("Uncheck all").clicked() {
                    for v in self.graph_selected_for_run.iter_mut() {
                        *v = false;
                    }
                }
            });

            left.add_space(8.0);
            left.separator();
            left.heading("Library");
            let selected_count = self.graph_selected_for_run.iter().filter(|&&b| b).count();
            left.label(
                RichText::new(format!(
                    "{} graphs in {} ({} checked for run)",
                    self.graphs.len(),
                    GRAPH_DIR,
                    selected_count
                ))
                .small()
                .weak(),
            );
            egui::ScrollArea::vertical().id_salt("graphs_scroll").show(left, |ui| {
                for i in 0..self.graphs.len() {
                    let id = self.graphs[i].spec.id();
                    let edges = self.graphs[i].edge_count;
                    let n = self.graphs[i].spec.n;
                    let is_preview = self.selected_graph == Some(i);
                    ui.horizontal(|ui| {
                        let mut sel = self.graph_selected_for_run[i];
                        if ui.checkbox(&mut sel, "").changed() {
                            self.graph_selected_for_run[i] = sel;
                        }
                        let label = format!("{} ({} edges, n={})", id, edges, n);
                        if ui.selectable_label(is_preview, label).clicked() {
                            self.selected_graph = Some(i);
                        }
                    });
                }
            });

            // Right: visualization
            let right = &mut cols[1];
            right.heading("Selected Graph (preview)");
            if let Some(g) = self.current_graph() {
                let info = format!(
                    "{}\n{} nodes, {} edges",
                    g.spec.id(),
                    g.spec.n,
                    g.edge_count
                );
                right.label(RichText::new(info).small().weak());
                right.add_space(4.0);
                draw_graph(right, g);
            } else {
                right.colored_label(Color32::GRAY, "No graph selected.");
            }
        });
    }

    fn tab_configs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Run Configurations (Sweeps)");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("+ Add").clicked() {
                    let id = self.next_sweep_id;
                    self.next_sweep_id += 1;
                    let sw = ConfigSweep {
                        name: format!("sweep #{}", id),
                        thetas: vec![Some(0.0)],
                        log10_iterations: vec![4],
                        smoothings: vec![SmoothingSpec::None],
                    };
                    self.sweep_inputs.push(SweepInputs::from_sweep(&sw));
                    self.sweeps.push(sw);
                    self.sweep_selected_for_run.push(true);
                }
            });
        });
        ui.label(
            RichText::new(
                "Each sweep expands to (Theta values × log10(iter) values × smoothing kinds × K values). \
                 Use comma-separated lists. \"T0\"/\"off\" in Theta means T=0 (greedy).",
            )
            .small()
            .weak(),
        );
        ui.add_space(4.0);

        self.ensure_sweep_selection_len();
        let mut remove: Option<usize> = None;

        egui::ScrollArea::vertical().id_salt("cfg_scroll").show(ui, |ui| {
            for idx in 0..self.sweeps.len() {
                let mut dirty = false;
                egui::Frame::NONE
                    .fill(Color32::from_rgb(34, 38, 48))
                    .corner_radius(CornerRadius::same(6))
                    .stroke(Stroke::new(1.0, Color32::from_rgb(60, 64, 76)))
                    .inner_margin(8.0)
                    .outer_margin(egui::Margin::symmetric(0, 3))
                    .show(ui, |ui| {
                        let sw = &mut self.sweeps[idx];
                        let inp = &mut self.sweep_inputs[idx];

                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            if ui
                                .add(egui::TextEdit::singleline(&mut sw.name).desired_width(180.0))
                                .changed()
                            {
                                dirty = true;
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.small_button("\u{2715}").on_hover_text("Remove").clicked() {
                                    remove = Some(idx);
                                }
                                ui.label(
                                    RichText::new(format!("\u{2192} {} configs", sw.count()))
                                        .small()
                                        .weak(),
                                );
                            });
                        });

                        ui.horizontal(|ui| {
                            ui.label("Theta values:")
                                .on_hover_text("Comma-separated, e.g. \"-1.0, 0.0, 1.0\". \"T0\" / \"off\" = T = 0 (greedy).");
                            if ui
                                .add(
                                    egui::TextEdit::singleline(&mut inp.thetas_text)
                                        .desired_width(ui.available_width() - 10.0)
                                        .hint_text("-1.0, 0.0, 1.0, T0"),
                                )
                                .changed()
                            {
                                dirty = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("log10(iter):")
                                .on_hover_text("Comma-separated integers. iterations = 10^N.");
                            if ui
                                .add(
                                    egui::TextEdit::singleline(&mut inp.iters_text)
                                        .desired_width(ui.available_width() - 10.0)
                                        .hint_text("3, 4, 5"),
                                )
                                .changed()
                            {
                                dirty = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Smoothing kinds:");
                            let labels = ["None", "K-Avg (det)", "K-Avg (rand)", "Weighted"];
                            for (i, lbl) in labels.iter().enumerate() {
                                if ui.checkbox(&mut inp.smoothing_kinds[i], *lbl).changed() {
                                    dirty = true;
                                }
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("K values:")
                                .on_hover_text("Comma-separated positive integers. Ignored if only \"None\" smoothing is selected.");
                            if ui
                                .add(
                                    egui::TextEdit::singleline(&mut inp.ks_text)
                                        .desired_width(ui.available_width() - 10.0)
                                        .hint_text("5, 10, 20"),
                                )
                                .changed()
                            {
                                dirty = true;
                            }
                        });

                        if let Some(err) = &inp.last_error {
                            ui.colored_label(
                                Color32::from_rgb(220, 100, 100),
                                RichText::new(format!("parse error: {}", err)).small(),
                            );
                        }
                    });

                if dirty {
                    let inp = &mut self.sweep_inputs[idx];
                    let sw = &mut self.sweeps[idx];
                    let _ = inp.apply_to_sweep(sw);
                }
            }
        });

        if let Some(i) = remove {
            self.sweeps.remove(i);
            self.sweep_inputs.remove(i);
            self.sweep_selected_for_run.remove(i);
        }
    }

    fn tab_run(&mut self, ui: &mut egui::Ui) {
        self.ensure_sweep_selection_len();
        ui.columns(2, |cols| {
            let left = &mut cols[0];
            left.heading("Target graphs");
            let selected_graphs = self.selected_graphs_for_run();
            if selected_graphs.is_empty() {
                left.colored_label(
                    Color32::GRAY,
                    "No graph checked (use the Graphs tab to check rows).",
                );
            } else {
                left.label(format!("{} graph(s) selected:", selected_graphs.len()));
                egui::ScrollArea::vertical()
                    .id_salt("run_graphs_scroll")
                    .max_height(120.0)
                    .show(left, |ui| {
                        for g in &selected_graphs {
                            ui.label(
                                RichText::new(format!(
                                    "  - {} ({} edges, n={})",
                                    g.spec.id(),
                                    g.edge_count,
                                    g.spec.n
                                ))
                                .small(),
                            );
                        }
                    });
            }

            left.add_space(8.0);
            left.heading("Sweeps to run");
            for i in 0..self.sweeps.len() {
                let cnt = self.sweeps[i].count();
                let label = format!(
                    "{}  (\u{2192} {} configs)",
                    self.sweeps[i].name, cnt
                );
                left.checkbox(&mut self.sweep_selected_for_run[i], label);
            }
            let expanded = self.expanded_selected_configs();
            left.label(
                RichText::new(format!("\u{2192} {} unique configs after expansion", expanded.len()))
                    .small()
                    .weak(),
            );

            left.add_space(8.0);
            left.separator();
            left.heading("Seeds & Threads");
            egui::Grid::new("seeds_grid").num_columns(2).spacing([8.0, 6.0]).show(left, |ui| {
                ui.label("Start seed:");
                ui.add(egui::DragValue::new(&mut self.start_seed).speed(1));
                ui.end_row();
                ui.label("# seeds:");
                ui.add(egui::Slider::new(&mut self.num_seeds, 1..=64));
                ui.end_row();
                ui.label("# threads:");
                ui.add(egui::Slider::new(&mut self.num_threads, 1..=self.max_threads.max(1)));
                ui.end_row();
            });
            left.label(
                RichText::new(format!(
                    "Seeds: {}..{}  | Logical cores: {}",
                    self.start_seed,
                    self.start_seed.wrapping_add(self.num_seeds as u64),
                    self.max_threads
                ))
                .small()
                .weak(),
            );
            let total_tasks = selected_graphs.len() * expanded.len() * self.num_seeds;
            left.label(
                RichText::new(format!("Total tasks: {}", total_tasks))
                    .small()
                    .weak(),
            );

            left.add_space(8.0);
            let in_progress = self.run_status.lock().unwrap().in_progress;
            left.horizontal(|ui| {
                if ui
                    .add_enabled(
                        !in_progress && !selected_graphs.is_empty() && !expanded.is_empty(),
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
            });

            // Right: progress + log
            let right = &mut cols[1];
            right.heading("Progress");
            let (in_progress, total, done, skipped, active, log) = {
                let st = self.run_status.lock().unwrap();
                (
                    st.in_progress,
                    st.total,
                    st.done,
                    st.skipped,
                    st.active_workers,
                    st.log.clone(),
                )
            };
            let pct = if total > 0 { done as f32 / total as f32 } else { 0.0 };
            right.add(egui::ProgressBar::new(pct).text(format!(
                "{}/{} (skipped {}, active {})",
                done, total, skipped, active
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
                if ui.button("Load matching").clicked() {
                    self.load_results_for_current();
                }
            });
        });
        ui.label(
            RichText::new(
                "Loads runs in data/results matching the currently selected graphs, sweeps, and seed range (Run tab).",
            )
            .small()
            .weak(),
        );
        ui.add_space(4.0);

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

        Plot::new("results_plot")
            .height(plot_h)
            .x_axis_label("log\u{2081}\u{2080}(step)")
            .y_axis_label("score")
            .legend(egui_plot::Legend::default())
            .show(ui, |pui| {
                let sel = self.selected_result;
                let to_plot: Vec<(usize, &RunResult)> = match sel {
                    Some(i) => self
                        .loaded_results
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j == i)
                        .collect(),
                    None => self.loaded_results.iter().enumerate().collect(),
                };
                for (_, r) in to_plot {
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
                                    "{} | {} | {} | s={} | {}",
                                    r.graph_spec.id(),
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
            });

        ui.add_space(6.0);
        ui.heading("Loaded runs");
        egui::ScrollArea::vertical().id_salt("results_scroll").show(ui, |ui| {
            if self.loaded_results.is_empty() {
                ui.colored_label(Color32::GRAY, "No results loaded.");
                return;
            }
            egui::Grid::new("results_grid")
                .striped(true)
                .min_col_width(60.0)
                .show(ui, |ui| {
                    ui.label(RichText::new("sel").strong());
                    ui.label(RichText::new("graph").strong());
                    ui.label(RichText::new("config").strong());
                    ui.label(RichText::new("seed").strong());
                    ui.label(RichText::new("steps").strong());
                    ui.label(RichText::new("ms").strong());
                    ui.label(RichText::new("final real").strong());
                    ui.end_row();
                    for (i, r) in self.loaded_results.iter().enumerate() {
                        let is_sel = self.selected_result == Some(i);
                        if ui
                            .selectable_label(is_sel, if is_sel { "\u{25C9}" } else { "\u{25CB}" })
                            .clicked()
                        {
                            self.selected_result = if is_sel { None } else { Some(i) };
                        }
                        ui.label(r.graph_spec.id());
                        ui.label(format!("{} ({})", r.config.name, r.config.id()));
                        ui.label(format!("{}", r.seed));
                        ui.label(format!("{}", r.records.len()));
                        ui.label(format!("{:.0}", r.elapsed_ms));
                        ui.label(format!(
                            "{:.2}",
                            r.records.last().map(|x| x.current_real).unwrap_or(f64::NAN)
                        ));
                        ui.end_row();
                    }
                });
        });
    }
}

fn trace_value(rec: &gpp_utils::run_executor::StepRecord, idx: usize) -> f64 {
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
