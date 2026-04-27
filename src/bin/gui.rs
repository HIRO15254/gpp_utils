//! GPP 実験用 GUI（4 タブ構成）。
//!
//! - Graphs: プリセット N/D/方式/シードでグラフを生成・永続化し、選択する。
//! - Configs: SA 実行条件（Θ、10^N 反復、スムージング）の集合を編集する。
//! - Run: 選択中のグラフと対象 Config 群、シード範囲で一括実行する（裏スレッド）。
//! - Results: 完了済み結果を 6 トレースの log-log プロットおよび TSV で確認する。

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

use eframe::egui;
use egui::{Color32, CornerRadius, RichText, Stroke};
use egui_plot::{Line, Plot, PlotPoints};

use gpp_utils::graph_spec::{
    EXPECTED_DEGREES, GraphKind, GraphLibrary, GraphSpec, NODE_COUNTS, StoredGraph,
};
use gpp_utils::run_config::{RunConfig, SmoothingSpec};
use gpp_utils::run_executor::{RunResult, ResultStore, execute};

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
            let mut style = (*cc.egui_ctx.style()).clone();
            style.spacing.item_spacing = egui::vec2(6.0, 4.0);
            style.spacing.slider_width = 140.0;
            cc.egui_ctx.set_style(style);
            Ok(Box::new(App::new()))
        }),
    )
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

struct App {
    library: GraphLibrary,
    store: ResultStore,

    // Graphs
    graphs: Vec<StoredGraph>,
    selected_graph: Option<usize>,
    new_kind: GraphKind,
    new_n_idx: usize,
    new_d_idx: usize,
    new_seed: u64,

    // Configs
    configs: Vec<RunConfig>,
    config_selected_for_run: Vec<bool>,
    next_config_id: usize,

    // Run params
    start_seed: u64,
    num_seeds: usize,

    // Run status (shared with thread)
    run_status: Arc<Mutex<RunStatus>>,

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

        let mut configs = Vec::new();
        configs.push(RunConfig {
            name: "T=1, 10^4".into(),
            theta: Some(0.0),
            log10_iterations: 4,
            smoothing: SmoothingSpec::None,
        });
        configs.push(RunConfig {
            name: "T=0, 10^4 (greedy)".into(),
            theta: None,
            log10_iterations: 4,
            smoothing: SmoothingSpec::None,
        });
        let config_selected_for_run = vec![true; configs.len()];

        let mut s = Self {
            library,
            store,
            selected_graph: if graphs.is_empty() { None } else { Some(0) },
            graphs,
            new_kind: GraphKind::Random,
            new_n_idx: 1,
            new_d_idx: 1,
            new_seed: 0,
            configs,
            config_selected_for_run,
            next_config_id: 3,
            start_seed: 0,
            num_seeds: 1,
            run_status: Arc::new(Mutex::new(RunStatus::default())),
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
        self.graphs = self.library.list();
        if let Some(i) = self.selected_graph {
            if i >= self.graphs.len() {
                self.selected_graph = if self.graphs.is_empty() { None } else { Some(0) };
            }
        } else if !self.graphs.is_empty() {
            self.selected_graph = Some(0);
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
                }
                self.status = format!("Graph ready: {}", spec.id());
            }
            Err(e) => self.status = format!("generate error: {}", e),
        }
    }

    fn start_run(&mut self) {
        let graph = match self.current_graph() {
            Some(g) => g.clone(),
            None => {
                self.status = "Select a graph first.".into();
                return;
            }
        };
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

        let total = cfgs.len() * self.num_seeds;
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
                cancel: false,
                log: vec![format!("Starting {} runs on {}", total, graph.spec.id())],
            };
        }

        let store_dir = self.store.base_dir.clone();
        let status_arc = Arc::clone(&self.run_status);
        let start_seed = self.start_seed;
        let num_seeds = self.num_seeds;

        thread::spawn(move || {
            let store = ResultStore::new(store_dir);
            let problem = graph.problem();
            'outer: for cfg in &cfgs {
                for s_off in 0..num_seeds {
                    {
                        let st = status_arc.lock().unwrap();
                        if st.cancel {
                            break 'outer;
                        }
                    }
                    let seed = start_seed.wrapping_add(s_off as u64);
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
                        continue;
                    }
                    let t0 = std::time::Instant::now();
                    let result = execute(graph.spec, cfg, &problem, seed);
                    let elapsed = t0.elapsed().as_secs_f64();
                    if let Err(e) = store.save(&result) {
                        let mut st = status_arc.lock().unwrap();
                        st.push_log(format!("save error: {}", e));
                    }
                    let mut st = status_arc.lock().unwrap();
                    st.done += 1;
                    st.push_log(format!(
                        "done {} / seed={} ({:.1}s, final real={:.2})",
                        cfg.id(),
                        seed,
                        elapsed,
                        result
                            .records
                            .last()
                            .map(|r| r.current_real)
                            .unwrap_or(f64::NAN)
                    ));
                }
            }
            let mut st = status_arc.lock().unwrap();
            st.in_progress = false;
            st.push_log("--- finished ---");
        });

        self.status = "Run started.".into();
    }

    fn cancel_run(&mut self) {
        let mut st = self.run_status.lock().unwrap();
        if st.in_progress {
            st.cancel = true;
            st.push_log("cancel requested");
        }
    }

    fn load_results_for_current(&mut self) {
        self.loaded_results.clear();
        self.selected_result = None;
        let graph = match self.current_graph() {
            Some(g) => g.clone(),
            None => {
                self.status = "Select a graph first.".into();
                return;
            }
        };
        let mut loaded = 0usize;
        for (i, cfg) in self.configs.iter().enumerate() {
            if !self.config_selected_for_run.get(i).copied().unwrap_or(false) {
                continue;
            }
            for s_off in 0..self.num_seeds {
                let seed = self.start_seed.wrapping_add(s_off as u64);
                if let Some(r) = self.store.load(&graph.spec, cfg, seed) {
                    self.loaded_results.push(r);
                    loaded += 1;
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
                RichText::new(format!("{} graphs in {}", self.graphs.len(), GRAPH_DIR))
                    .small()
                    .weak(),
            );
            egui::ScrollArea::vertical().id_salt("graphs_scroll").show(left, |ui| {
                for i in 0..self.graphs.len() {
                    let id = self.graphs[i].spec.id();
                    let edges = self.graphs[i].edge_count;
                    let n = self.graphs[i].spec.n;
                    let selected = self.selected_graph == Some(i);
                    let label = format!("{} ({} edges, n={})", id, edges, n);
                    if ui.selectable_label(selected, label).clicked() {
                        self.selected_graph = Some(i);
                    }
                }
            });

            // Right: visualization
            let right = &mut cols[1];
            right.heading("Selected Graph");
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

        self.ensure_config_selection_len();
        let mut remove: Option<usize> = None;
        let mut config_changed = false;

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
                                ui.label(RichText::new(cfg.id()).small().weak());
                            });
                        });

                        ui.horizontal(|ui| {
                            let mut has_theta = cfg.theta.is_some();
                            if ui
                                .checkbox(&mut has_theta, "use Theta")
                                .on_hover_text("If unchecked, T = 0 (no acceptance of worse moves)")
                                .changed()
                            {
                                cfg.theta = if has_theta { Some(0.0) } else { None };
                                config_changed = true;
                            }
                            if let Some(t) = &mut cfg.theta {
                                let resp = ui.add(
                                    egui::Slider::new(t, -3.0..=3.0)
                                        .text("Theta = log10(T)")
                                        .step_by(0.1),
                                );
                                if resp.changed() {
                                    config_changed = true;
                                }
                            } else {
                                ui.colored_label(Color32::GRAY, "T = 0");
                            }
                        });

                        ui.horizontal(|ui| {
                            let mut n = cfg.log10_iterations as i32;
                            if ui
                                .add(egui::Slider::new(&mut n, 1..=8).text("log10(iter)"))
                                .changed()
                            {
                                cfg.log10_iterations = n.max(0) as u32;
                                config_changed = true;
                            }
                            ui.label(format!("= {} iterations", cfg.iterations()));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Smoothing:");
                            let label = match cfg.smoothing {
                                SmoothingSpec::None => "None".to_string(),
                                SmoothingSpec::KAverage(k) => format!("K-Avg (det) K={}", k),
                                SmoothingSpec::RandomKAverage(k) => format!("K-Avg (rand) K={}", k),
                                SmoothingSpec::WeightedAverage(k) => format!("Weighted K={}", k),
                            };
                            egui::ComboBox::from_id_salt(format!("sm_{}", idx))
                                .selected_text(label)
                                .show_ui(ui, |ui| {
                                    let cur_k = match cfg.smoothing {
                                        SmoothingSpec::None => 8,
                                        SmoothingSpec::KAverage(k)
                                        | SmoothingSpec::RandomKAverage(k)
                                        | SmoothingSpec::WeightedAverage(k) => k,
                                    };
                                    if ui
                                        .selectable_label(
                                            matches!(cfg.smoothing, SmoothingSpec::None),
                                            "None",
                                        )
                                        .clicked()
                                    {
                                        cfg.smoothing = SmoothingSpec::None;
                                        config_changed = true;
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(cfg.smoothing, SmoothingSpec::KAverage(_)),
                                            "K-Avg (det)",
                                        )
                                        .clicked()
                                    {
                                        cfg.smoothing = SmoothingSpec::KAverage(cur_k);
                                        config_changed = true;
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(cfg.smoothing, SmoothingSpec::RandomKAverage(_)),
                                            "K-Avg (rand)",
                                        )
                                        .clicked()
                                    {
                                        cfg.smoothing = SmoothingSpec::RandomKAverage(cur_k);
                                        config_changed = true;
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(cfg.smoothing, SmoothingSpec::WeightedAverage(_)),
                                            "Weighted",
                                        )
                                        .clicked()
                                    {
                                        cfg.smoothing = SmoothingSpec::WeightedAverage(cur_k);
                                        config_changed = true;
                                    }
                                });
                            match &mut cfg.smoothing {
                                SmoothingSpec::None => {}
                                SmoothingSpec::KAverage(k)
                                | SmoothingSpec::RandomKAverage(k)
                                | SmoothingSpec::WeightedAverage(k) => {
                                    if ui
                                        .add(egui::Slider::new(k, 1..=64).text("K"))
                                        .changed()
                                    {
                                        config_changed = true;
                                    }
                                }
                            }
                        });
                    });
            }
        });

        if let Some(i) = remove {
            self.configs.remove(i);
            self.config_selected_for_run.remove(i);
        }
        if config_changed {
            // no-op marker for future use
        }
    }

    fn tab_run(&mut self, ui: &mut egui::Ui) {
        self.ensure_config_selection_len();
        ui.columns(2, |cols| {
            let left = &mut cols[0];
            left.heading("Target");
            if let Some(g) = self.current_graph() {
                left.label(format!("Graph: {}", g.spec.id()));
                left.label(format!("  ({} nodes, {} edges)", g.spec.n, g.edge_count));
            } else {
                left.colored_label(Color32::GRAY, "No graph selected (Graphs tab).");
            }
            left.add_space(8.0);
            left.label("Configs to run:");
            for i in 0..self.configs.len() {
                let cfg_label = format!("{}  ({})", self.configs[i].name, self.configs[i].id());
                left.checkbox(&mut self.config_selected_for_run[i], cfg_label);
            }

            left.add_space(8.0);
            left.separator();
            left.heading("Seeds");
            egui::Grid::new("seeds_grid").num_columns(2).spacing([8.0, 6.0]).show(left, |ui| {
                ui.label("Start seed:");
                ui.add(egui::DragValue::new(&mut self.start_seed).speed(1));
                ui.end_row();
                ui.label("# seeds:");
                ui.add(egui::Slider::new(&mut self.num_seeds, 1..=64));
                ui.end_row();
            });
            left.label(
                RichText::new(format!(
                    "Seeds: {}..{}",
                    self.start_seed,
                    self.start_seed.wrapping_add(self.num_seeds as u64)
                ))
                .small()
                .weak(),
            );

            left.add_space(8.0);
            let in_progress = self.run_status.lock().unwrap().in_progress;
            left.horizontal(|ui| {
                if ui
                    .add_enabled(
                        !in_progress && self.current_graph().is_some(),
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
                if ui.button("Load matching").clicked() {
                    self.load_results_for_current();
                }
            });
        });
        ui.label(
            RichText::new(
                "Loads runs in data/results matching the currently selected graph, configs, and seed range (Run tab).",
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
