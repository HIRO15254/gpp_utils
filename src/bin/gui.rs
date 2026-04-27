use eframe::egui;
use egui::{Color32, CornerRadius, RichText, Stroke};
use egui_plot::{Line, Plot, PlotPoints};
use rand_mt::Mt19937GenRand64;
use std::time::Instant;

use gpp_utils::graph_partition::{Graph, GraphGenerationMethod, GraphPartitionProblem};
use gpp_utils::optimization::{Problem, Solver};
use gpp_utils::smoothing::{KAveragingSmoothing, NoSmoothing};
use gpp_utils::solvers::{
    ExtremalOptimizationSolver, HillClimbingSolver,
    SimulatedAnnealingSolver, SimulatedQuantumAnnealingSolver,
};

// ---------------------------------------------------------------------------
// Color palette
// ---------------------------------------------------------------------------
const PALETTE: &[Color32] = &[
    Color32::from_rgb(56, 132, 212),
    Color32::from_rgb(228, 108, 10),
    Color32::from_rgb(55, 168, 75),
    Color32::from_rgb(204, 51, 63),
    Color32::from_rgb(142, 99, 186),
    Color32::from_rgb(237, 177, 32),
    Color32::from_rgb(0, 170, 160),
    Color32::from_rgb(200, 82, 148),
    Color32::from_rgb(107, 76, 54),
    Color32::from_rgb(128, 128, 128),
];

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1320.0, 880.0])
            .with_title("GPP Solver Comparison"),
        ..Default::default()
    };
    eframe::run_native(
        "GPP Solver Comparison",
        options,
        Box::new(|cc| {
            let mut style = (*cc.egui_ctx.style()).clone();
            style.spacing.item_spacing = egui::vec2(6.0, 4.0);
            style.spacing.slider_width = 140.0;
            cc.egui_ctx.set_style(style);
            Ok(Box::new(SolverApp::default()))
        }),
    )
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq)]
enum GenMethod { Random, Geometric }

#[derive(Debug, Clone, Copy, PartialEq)]
enum SolverKind { HC, SA, EO, SQA }

impl SolverKind {
    const ALL: &[Self] = &[Self::HC, Self::SA, Self::EO, Self::SQA];

    fn label(self) -> &'static str {
        match self {
            Self::HC => "HC",
            Self::SA => "SA",
            Self::EO => "EO",
            Self::SQA => "SQA",
        }
    }

    fn tip(self) -> &'static str {
        match self {
            Self::HC => "Hill Climbing: greedy local search to nearest local optimum",
            Self::SA  => "Simulated Annealing: Metropolis acceptance with fixed temperature",
            Self::EO  => "Extremal Optimization: τ-EO, mutates worst-fitness components",
            Self::SQA => "Simulated Quantum Annealing: replica-based quantum tunneling",
        }
    }

    fn supports_smoothing(self) -> bool {
        self != SolverKind::SQA
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SmoothingKind { None, KAverage }

impl SmoothingKind {
    fn label(self) -> &'static str {
        match self {
            Self::None     => "None",
            Self::KAverage => "K-Average",
        }
    }
}

#[derive(Debug, Clone)]
struct SolverEntry {
    kind: SolverKind,
    label: String,
    enabled: bool,
    // Smoothing (for HC, SA, EO)
    smoothing_kind: SmoothingKind,
    smoothing_k: usize,
    // SA params
    sa_temperature: f64,
    sa_iterations: usize,
    // EO params
    eo_tau: f64,
    eo_iterations: usize,
    // SQA params
    sqa_replicas: usize,
    sqa_temperature: f64,
    sqa_gamma_init: f64,
    sqa_gamma_final: f64,
    sqa_steps: usize,
}

impl SolverEntry {
    fn new(kind: SolverKind, id: usize) -> Self {
        Self {
            kind,
            label: format!("{} #{}", kind.label(), id),
            enabled: true,
            smoothing_kind: SmoothingKind::None,
            smoothing_k: 10,
            sa_temperature: 10.0,
            sa_iterations: 50_000,
            eo_tau: 1.5,
            eo_iterations: 50_000,
            sqa_replicas: 16,
            sqa_temperature: 0.1,
            sqa_gamma_init: 5.0,
            sqa_gamma_final: 0.01,
            sqa_steps: 500,
        }
    }
}

struct SolverResult {
    label: String,
    color: Color32,
    score_history: Vec<[f64; 2]>,
    final_partition: Vec<bool>,
    best_score: f64,
    elapsed_ms: f64,
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------
struct SolverApp {
    node_count: usize,
    expected_degree: f64,
    gen_method: GenMethod,
    seed: u64,
    problem: Option<GraphPartitionProblem>,
    coordinates: Option<Vec<(f64, f64)>>,
    graph_info: String,
    entries: Vec<SolverEntry>,
    add_kind: SolverKind,
    next_id: usize,
    results: Vec<SolverResult>,
    selected_result: Option<usize>,
    status: String,
}

impl Default for SolverApp {
    fn default() -> Self {
        let mut app = Self {
            node_count: 50,
            expected_degree: 5.0,
            gen_method: GenMethod::Geometric,
            seed: 42,
            problem: None,
            coordinates: None,
            graph_info: String::new(),
            entries: Vec::new(),
            add_kind: SolverKind::SA,
            next_id: 1,
            results: Vec::new(),
            selected_result: None,
            status: "Generate a graph, add solvers, then run.".into(),
        };
        app.entries.push(SolverEntry::new(SolverKind::SA, 1));
        app.entries.push(SolverEntry::new(SolverKind::EO, 2));
        app.next_id = 3;
        app
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn tip_slider_f64(
    ui: &mut egui::Ui, v: &mut f64,
    r: std::ops::RangeInclusive<f64>, name: &str, tip: &str, log: bool,
) {
    let mut s = egui::Slider::new(v, r).text(name);
    if log { s = s.logarithmic(true); }
    ui.add(s).on_hover_text(tip);
}

fn tip_slider_usize(
    ui: &mut egui::Ui, v: &mut usize,
    r: std::ops::RangeInclusive<usize>, name: &str, tip: &str, log: bool,
) {
    let mut s = egui::Slider::new(v, r).text(name);
    if log { s = s.logarithmic(true); }
    ui.add(s).on_hover_text(tip);
}

fn smoothing_ui(ui: &mut egui::Ui, e: &mut SolverEntry) {
    if !e.kind.supports_smoothing() {
        ui.label(RichText::new("Smoothing: built-in (quantum tunneling)").small().weak());
        return;
    }
    ui.horizontal(|ui| {
        ui.label("Smoothing:");
        egui::ComboBox::from_id_salt(format!("sm_{}", e.label))
            .width(90.0)
            .selected_text(e.smoothing_kind.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut e.smoothing_kind, SmoothingKind::None, "None")
                    .on_hover_text("Use raw problem score (no smoothing)");
                ui.selectable_value(&mut e.smoothing_kind, SmoothingKind::KAverage, "K-Average")
                    .on_hover_text("Average score over K random neighbours");
            });
        if e.smoothing_kind == SmoothingKind::KAverage {
            tip_slider_usize(ui, &mut e.smoothing_k, 2..=200, "K",
                "Number of neighbours to average", false);
        }
    });
}

fn solver_params_ui(ui: &mut egui::Ui, e: &mut SolverEntry) {
    smoothing_ui(ui, e);
    match e.kind {
        SolverKind::HC => {}
        SolverKind::SA => {
            tip_slider_f64(ui, &mut e.sa_temperature, 0.1..=100.0, "T",
                "Temperature: higher = more exploration, lower = greedy", true);
            tip_slider_usize(ui, &mut e.sa_iterations, 1_000..=1_000_000, "Iterations",
                "Total number of flip attempts", true);
        }
        SolverKind::EO => {
            tip_slider_f64(ui, &mut e.eo_tau, 1.0..=5.0, "\u{03C4}",
                "Power-law exponent. Larger = always picks worst component", false);
            tip_slider_usize(ui, &mut e.eo_iterations, 1_000..=1_000_000, "Iterations",
                "Total number of EO steps", true);
        }
        SolverKind::SQA => {
            tip_slider_usize(ui, &mut e.sqa_replicas, 2..=64, "P",
                "Trotter replicas: more = better tunneling, slower", false);
            tip_slider_f64(ui, &mut e.sqa_temperature, 0.01..=10.0, "T",
                "Thermal temperature (fixed) for Metropolis", true);
            tip_slider_f64(ui, &mut e.sqa_gamma_init, 0.1..=100.0, "\u{0393}i",
                "Initial transverse field: larger = stronger quantum fluctuations", true);
            tip_slider_f64(ui, &mut e.sqa_gamma_final, 0.0001..=1.0, "\u{0393}f",
                "Final transverse field: smaller = replicas converge", true);
            tip_slider_usize(ui, &mut e.sqa_steps, 100..=50_000, "Steps",
                "MC steps (each = P\u{00D7}n flip attempts)", true);
        }
    }
}

// ---------------------------------------------------------------------------
// Solver execution
// ---------------------------------------------------------------------------
fn run_entry(e: &SolverEntry, prob: &GraphPartitionProblem, seed: u64, col: Color32) -> SolverResult {
    let t0 = Instant::now();
    let mut r = Mt19937GenRand64::new(seed);
    let ini = prob.random_solution(&mut r);

    let (sol, stats) = if e.kind == SolverKind::SQA {
        let sqa = SimulatedQuantumAnnealingSolver::new(
            e.sqa_replicas, e.sqa_temperature,
            e.sqa_gamma_init, e.sqa_gamma_final, e.sqa_steps,
        );
        sqa.solve(prob, ini, &mut r)
    } else {
        match e.smoothing_kind {
            SmoothingKind::None     => dispatch_solver(e, prob, ini, &mut r, &NoSmoothing),
            SmoothingKind::KAverage => dispatch_solver(e, prob, ini, &mut r, &KAveragingSmoothing::new(e.smoothing_k)),
        }
    };

    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let h: Vec<[f64; 2]> = stats.score_history.iter().map(|&(i, s)| [i as f64, s]).collect();
    SolverResult {
        label: e.label.clone(),
        color: col,
        score_history: h,
        final_partition: sol,
        best_score: stats.best_score,
        elapsed_ms: ms,
    }
}

fn dispatch_solver<Sm>(
    e: &SolverEntry,
    prob: &GraphPartitionProblem,
    ini: Vec<bool>,
    r: &mut Mt19937GenRand64,
    smoothing: &Sm,
) -> (Vec<bool>, gpp_utils::optimization::SolverStats)
where
    Sm: gpp_utils::optimization::Smoothing<Vec<bool>>,
{
    match e.kind {
        SolverKind::HC => HillClimbingSolver::new().solve(prob, smoothing, ini, r),
        SolverKind::SA => SimulatedAnnealingSolver::new(e.sa_temperature, e.sa_iterations)
            .solve(prob, smoothing, ini, r),
        SolverKind::EO => ExtremalOptimizationSolver::new(Some(e.eo_tau), e.eo_iterations)
            .solve(prob, smoothing, ini, r),
        SolverKind::SQA => unreachable!("SQA is handled separately"),
    }
}

// ---------------------------------------------------------------------------
// App methods
// ---------------------------------------------------------------------------
impl SolverApp {
    fn generate_graph(&mut self) {
        let mut rng = Mt19937GenRand64::new(self.seed);
        let (problem, coords) = match self.gen_method {
            GenMethod::Random => {
                let m = GraphGenerationMethod::Random {
                    node_count: self.node_count,
                    expected_degree: self.expected_degree,
                };
                let prob = GraphPartitionProblem::generate(m, &mut rng);
                let c: Vec<(f64, f64)> = (0..self.node_count).map(|i| {
                    let a = 2.0 * std::f64::consts::PI * i as f64 / self.node_count as f64;
                    (0.5 + 0.45 * a.cos(), 0.5 + 0.45 * a.sin())
                }).collect();
                (prob, c)
            }
            GenMethod::Geometric => {
                GraphPartitionProblem::generate_geometric_with_coords(
                    self.node_count, self.expected_degree, &mut rng,
                )
            }
        };

        let g = problem.graph();
        let edges = g.adjacency_list.iter().map(|a| a.len()).sum::<usize>() / 2;
        self.graph_info = format!("{} nodes, {} edges", g.node_count, edges);
        self.coordinates = Some(coords);
        self.problem = Some(problem);
        self.results.clear();
        self.selected_result = None;
        self.status = format!("Graph: {}", self.graph_info);
    }

    fn run_solvers(&mut self) {
        let prob = match &self.problem {
            Some(p) => p.clone(),
            None => { self.status = "Generate a graph first.".into(); return; }
        };
        self.results.clear();
        self.selected_result = None;
        for (i, e) in self.entries.iter().enumerate() {
            if !e.enabled { continue; }
            self.results.push(run_entry(e, &prob, self.seed, PALETTE[i % PALETTE.len()]));
        }
        if self.results.is_empty() {
            self.status = "No enabled solvers.".into();
        } else {
            self.selected_result = Some(0);
            let b = self.results.iter()
                .min_by(|a, b| a.best_score.partial_cmp(&b.best_score).unwrap())
                .unwrap();
            self.status = format!("Best: {} = {:.2}", b.label, b.best_score);
        }
    }

    fn draw_graph(&self, ui: &mut egui::Ui) {
        let coords = match &self.coordinates {
            Some(c) => c,
            None => { ui.colored_label(Color32::GRAY, "No graph."); return; }
        };
        let graph: &Graph = self.problem.as_ref().unwrap().graph();
        let part = self.selected_result
            .and_then(|i| self.results.get(i))
            .map(|r| &r.final_partition);

        let av = ui.available_size();
        let sz = av.x.min(av.y);
        let (resp, painter) = ui.allocate_painter(egui::vec2(sz, sz), egui::Sense::hover());
        let rect = resp.rect;
        painter.rect_filled(rect, CornerRadius::same(4), Color32::from_rgb(24, 26, 32));
        let inner = rect.shrink(10.0);
        let to_s = |x: f64, y: f64| egui::pos2(
            inner.left() + x as f32 * inner.width(),
            inner.top() + (1.0 - y) as f32 * inner.height(),
        );

        let ecol = Color32::from_rgba_premultiplied(100, 110, 130, 50);
        for u in 0..graph.node_count {
            for &v in &graph.adjacency_list[u] {
                if u < v {
                    painter.line_segment(
                        [to_s(coords[u].0, coords[u].1), to_s(coords[v].0, coords[v].1)],
                        Stroke::new(0.6, ecol),
                    );
                }
            }
        }

        let r = (3.5_f32).max(70.0 / (graph.node_count as f32).sqrt());
        for i in 0..graph.node_count {
            let p = to_s(coords[i].0, coords[i].1);
            let fill = match part {
                Some(pa) if pa[i] => Color32::from_rgb(56, 152, 232),
                Some(_)           => Color32::from_rgb(232, 72, 85),
                None              => Color32::from_rgb(160, 160, 170),
            };
            painter.circle(p, r, fill, Stroke::new(0.8, Color32::from_rgb(40, 42, 50)));
        }
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------
impl eframe::App for SolverApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ---- Left panel ----
        egui::SidePanel::left("left").min_width(310.0).max_width(380.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(4.0);
                ui.heading("Graph Generation");
                ui.add_space(2.0);
                egui::Grid::new("g_grid").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
                    ui.label("Nodes:");
                    ui.add(egui::Slider::new(&mut self.node_count, 4..=500))
                        .on_hover_text("Number of vertices");
                    ui.end_row();
                    ui.label("Exp. degree:");
                    ui.add(egui::Slider::new(&mut self.expected_degree, 1.0..=30.0))
                        .on_hover_text("Expected neighbours per vertex");
                    ui.end_row();
                    ui.label("Method:");
                    egui::ComboBox::from_id_salt("gm")
                        .selected_text(match self.gen_method {
                            GenMethod::Random => "Erdos-Renyi",
                            GenMethod::Geometric => "Geometric",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.gen_method, GenMethod::Random, "Erdos-Renyi")
                                .on_hover_text("G(n,p): edges exist independently with prob p");
                            ui.selectable_value(&mut self.gen_method, GenMethod::Geometric, "Geometric")
                                .on_hover_text("Random points in [0,1]\u{00B2}; nearby points connected");
                        });
                    ui.end_row();
                    ui.label("Seed:");
                    ui.add(egui::DragValue::new(&mut self.seed).speed(1));
                    ui.end_row();
                });
                ui.add_space(4.0);
                if ui.add_sized(
                    [ui.available_width(), 28.0],
                    egui::Button::new(RichText::new("Generate Graph").strong()),
                ).clicked() {
                    self.generate_graph();
                }
                if !self.graph_info.is_empty() {
                    ui.label(RichText::new(&self.graph_info).small().weak());
                }
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                // Solver entries
                ui.horizontal(|ui| {
                    ui.heading("Solver Runs");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("+ Add").clicked() {
                            self.entries.push(SolverEntry::new(self.add_kind, self.next_id));
                            self.next_id += 1;
                        }
                        egui::ComboBox::from_id_salt("ak").width(80.0)
                            .selected_text(self.add_kind.label())
                            .show_ui(ui, |ui| {
                                for &k in SolverKind::ALL {
                                    ui.selectable_value(&mut self.add_kind, k, k.label())
                                        .on_hover_text(k.tip());
                                }
                            });
                    });
                });
                ui.add_space(2.0);

                let mut rm = None;
                for (idx, entry) in self.entries.iter_mut().enumerate() {
                    let bg = if entry.enabled {
                        Color32::from_rgb(38, 42, 52)
                    } else {
                        Color32::from_rgb(30, 30, 36)
                    };
                    egui::Frame::NONE
                        .fill(bg)
                        .corner_radius(CornerRadius::same(6))
                        .inner_margin(8.0)
                        .outer_margin(egui::Margin::symmetric(0, 2))
                        .stroke(Stroke::new(1.0, Color32::from_rgb(55, 60, 72)))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut entry.enabled, "");
                                let c = PALETTE[idx % PALETTE.len()];
                                ui.colored_label(c, RichText::new(&entry.label).strong());
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button("\u{2715}").on_hover_text("Remove").clicked() {
                                        rm = Some(idx);
                                    }
                                    ui.add(egui::TextEdit::singleline(&mut entry.label).desired_width(80.0))
                                        .on_hover_text("Rename this run");
                                });
                            });
                            ui.label(RichText::new(entry.kind.tip()).small().weak());
                            ui.add_space(2.0);
                            solver_params_ui(ui, entry);
                        });
                }
                if let Some(i) = rm { self.entries.remove(i); }

                ui.add_space(8.0);
                if ui.add_enabled(
                    self.problem.is_some(),
                    egui::Button::new(RichText::new("Run All").strong().size(15.0))
                        .min_size(egui::vec2(ui.available_width(), 32.0)),
                ).clicked() {
                    self.run_solvers();
                }
                ui.add_space(4.0);
                ui.label(RichText::new(&self.status).italics());
                ui.add_space(8.0);
            });
        });

        // ---- Central panel ----
        egui::CentralPanel::default().show(ctx, |ui| {
            let plot_h = (ui.available_height() * 0.50).max(200.0);
            ui.horizontal(|ui| {
                ui.heading("Score Progression");
                ui.label(RichText::new("(log-log)").small().weak());
            });
            Plot::new("score_plot")
                .height(plot_h)
                .x_axis_label("log\u{2081}\u{2080}(Iteration)")
                .y_axis_label("log\u{2081}\u{2080}(Best Score)")
                .legend(egui_plot::Legend::default())
                .show(ui, |pui| {
                    for r in &self.results {
                        let d: Vec<[f64; 2]> = r.score_history.iter()
                            .filter(|p| p[0] > 0.0 && p[1] > 0.0)
                            .map(|p| [p[0].log10(), p[1].log10()])
                            .collect();
                        pui.line(Line::new(PlotPoints::new(d)).name(&r.label).color(r.color).width(2.0));
                    }
                });
            ui.add_space(4.0);

            ui.columns(2, |cols| {
                // Results table
                cols[0].heading("Results");
                if self.results.is_empty() {
                    cols[0].colored_label(Color32::GRAY, "No results yet.");
                } else {
                    egui::ScrollArea::vertical().id_salt("rs").show(&mut cols[0], |ui| {
                        egui::Grid::new("rg").striped(true).min_col_width(50.0).show(ui, |ui| {
                            ui.label(RichText::new("Run").strong());
                            ui.label(RichText::new("Best").strong());
                            ui.label(RichText::new("ms").strong());
                            ui.label("");
                            ui.end_row();
                            for (i, r) in self.results.iter().enumerate() {
                                let sel = self.selected_result == Some(i);
                                ui.colored_label(r.color, &r.label);
                                ui.label(format!("{:.2}", r.best_score));
                                ui.label(format!("{:.0}", r.elapsed_ms));
                                if ui.selectable_label(
                                    sel,
                                    if sel { "\u{25C9}" } else { "\u{25CB}" },
                                ).on_hover_text("Show partition").clicked() {
                                    self.selected_result = Some(i);
                                }
                                ui.end_row();
                            }
                        });
                    });
                }

                // Graph viz
                cols[1].heading("Graph Partition");
                self.draw_graph(&mut cols[1]);
            });
        });
    }
}
