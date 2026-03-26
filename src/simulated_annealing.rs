use crate::optimization::OptimizationProblem;
use rand::Rng;
use rand_mt::Mt19937GenRand64;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fs;
use std::path::Path;

/// Cooling schedule strategies for simulated annealing
#[derive(Debug, Clone, Copy)]
pub enum CoolingSchedule {
    /// Constant temperature: T = temperature
    Constant { temperature: f64 },
}

/// Configuration parameters for simulated annealing
#[derive(Debug, Clone)]
pub struct SimulatedAnnealingConfig {
    /// Cooling schedule strategy
    pub cooling_schedule: CoolingSchedule,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Logging interval (None = no logging)
    pub log_interval: Option<usize>,
}

impl Default for SimulatedAnnealingConfig {
    fn default() -> Self {
        Self {
            cooling_schedule: CoolingSchedule::Constant { temperature: 10.0 },
            max_iterations: 10000,
            log_interval: None,
        }
    }
}

impl SimulatedAnnealingConfig {
    /// Create a new configuration with constant temperature
    pub fn new_constant(temperature: f64, max_iterations: usize) -> Self {
        Self {
            cooling_schedule: CoolingSchedule::Constant { temperature },
            max_iterations,
            log_interval: None,
        }
    }

    /// Set logging interval
    pub fn with_logging(mut self, interval: usize) -> Self {
        self.log_interval = Some(interval);
        self
    }
}

/// Statistics collected during simulated annealing
#[derive(Debug, Clone)]
pub struct AnnealingStats {
    pub iterations_completed: usize,
    pub final_temperature: f64,
    pub initial_score: f64,
    pub final_score: f64,
    pub best_score: f64,
    pub accepted_moves: usize,
    pub rejected_moves: usize,
    pub score_history: Vec<(usize, f64)>,
}

/// Generic simulated annealing solver
/// 
/// # Example
/// ```
/// use gpp_utils::simulated_annealing::{SimulatedAnnealingSolver, SimulatedAnnealingConfig};
/// use gpp_utils::graph_partition::{Graph, GraphPartitionProblem};
/// use gpp_utils::optimization::OptimizationProblem;
/// use rand_mt::Mt19937GenRand64;
/// 
/// // Create a problem
/// let mut graph = Graph::new(4);
/// graph.add_edge(0, 1);
/// graph.add_edge(1, 2);
/// let problem = GraphPartitionProblem::new(graph);
/// 
/// // Configure solver
/// let config = SimulatedAnnealingConfig::new_constant(10.0, 1000)
///     .with_logging(100);
/// let solver = SimulatedAnnealingSolver::new(config);
/// 
/// // Solve
/// let initial_solution = vec![true, false, true, false];
/// let mut rng = Mt19937GenRand64::new(42);
/// let (solution, stats) = solver.solve(&problem, initial_solution, &mut rng);
/// ```
pub struct SimulatedAnnealingSolver {
    config: SimulatedAnnealingConfig,
}

impl SimulatedAnnealingSolver {
    /// Create a new solver with the given configuration
    pub fn new(config: SimulatedAnnealingConfig) -> Self {
        Self { config }
    }

    /// Create a solver with default configuration
    pub fn default() -> Self {
        Self::new(SimulatedAnnealingConfig::default())
    }

    /// Create a solver with quick constant temperature setup
    pub fn constant(temperature: f64, max_iterations: usize) -> Self {
        Self::new(SimulatedAnnealingConfig::new_constant(temperature, max_iterations))
    }

    /// Solve the optimization problem using simulated annealing
    pub fn solve<P, S: Clone, G>(
        &self,
        problem: &dyn OptimizationProblem<P, S, G>,
        initial_solution: S,
        rng: &mut Mt19937GenRand64,
    ) -> (S, AnnealingStats) {
        let mut current_solution = initial_solution;
        let mut current_score = problem.score(&current_solution);
        let mut best_solution = current_solution.clone();
        let mut best_score = current_score;
        let initial_score = current_score;

        let temperature = self.calculate_temperature(0);
        let mut accepted_moves = 0;
        let mut rejected_moves = 0;
        let mut iterations_completed = 0;

        let mut score_history = Vec::new();
        let record_interval = (self.config.max_iterations / 100).max(1);

        for iteration in 0..self.config.max_iterations {
            // Generate random neighbor
            let (neighbor_solution, neighbor_score) = 
                problem.random_neighbour(&current_solution, current_score, rng);

            // Calculate acceptance probability
            let accept = if neighbor_score < current_score {
                true // Always accept better solutions (minimization)
            } else if temperature > 0.0 {
                let delta = current_score - neighbor_score; // Reversed for minimization
                let probability = (delta / temperature).exp();
                rng.r#gen::<f64>() < probability
            } else {
                false
            };

            if accept {
                current_solution = neighbor_solution;
                current_score = neighbor_score;
                accepted_moves += 1;

                // Update the best solution if current is better (minimization)
                if current_score < best_score {
                    best_solution = current_solution.clone();
                    best_score = current_score;
                }
            } else {
                rejected_moves += 1;
            }

            if iteration % record_interval == 0 {
                score_history.push((iteration, best_score));
            }

            // Optional logging
            if let Some(log_interval) = self.config.log_interval {
                if iteration % log_interval == 0 {
                    println!(
                        "Iteration {}: T={:.6}, Current={:.6}, Best={:.6}, Accept%={:.2}%",
                        iteration,
                        temperature,
                        current_score,
                        best_score,
                        (accepted_moves as f64 / (accepted_moves + rejected_moves) as f64) * 100.0
                    );
                }
            }


            iterations_completed = iteration + 1;
        }

        score_history.push((iterations_completed, best_score));

        let stats = AnnealingStats {
            iterations_completed,
            final_temperature: temperature,
            initial_score,
            final_score: current_score,
            best_score,
            accepted_moves,
            rejected_moves,
            score_history,
        };

        (best_solution, stats)
    }

    /// Calculate temperature at given iteration according to cooling schedule
    fn calculate_temperature(&self, _iteration: usize) -> f64 {
        match self.config.cooling_schedule {
            CoolingSchedule::Constant { temperature } => temperature,
        }
    }
}

// ============================================================================
// Adaptive Simulated Annealing (ASA) Implementation
// ============================================================================

/// Configuration parameters for Adaptive Simulated Annealing
#[derive(Debug, Clone)]
pub struct AdaptiveSAConfig {
    /// Initial acceptance probability (ps)
    pub initial_acceptance_probability: f64,
    /// Quasi-equilibrium constant (ε)
    pub epsilon: f64,
    /// Cooling ratio (γ)
    pub gamma: f64,
    /// Number of parallel processes (I)
    pub num_processes: usize,
    /// Maximum number of total exploration steps
    pub max_steps: usize,
    /// Logging interval (None = no logging)
    pub log_interval: Option<usize>,
    /// Recording configuration (None = no recording)
    pub recording: Option<RecordingConfig>,
    /// Pre-calculated initial temperature (None = calculate from ps)
    pub initial_temperature: Option<f64>,
}

impl Default for AdaptiveSAConfig {
    fn default() -> Self {
        Self {
            initial_acceptance_probability: 0.5,
            epsilon: 0.3,
            gamma: 0.75,
            num_processes: 8,
            max_steps: 1000000,
            log_interval: None,
            recording: None,
            initial_temperature: None,
        }
    }
}

impl AdaptiveSAConfig {
    /// Create a new ASA configuration
    pub fn new(ps: f64, epsilon: f64, gamma: f64, num_processes: usize, max_steps: usize) -> Self {
        Self {
            initial_acceptance_probability: ps,
            epsilon,
            gamma,
            num_processes,
            max_steps,
            log_interval: None,
            recording: None,
            initial_temperature: None,
        }
    }

    /// Set logging interval
    pub fn with_logging(mut self, interval: usize) -> Self {
        self.log_interval = Some(interval);
        self
    }

    /// Set recording configuration
    pub fn with_recording(mut self, recording: RecordingConfig) -> Self {
        self.recording = Some(recording);
        self
    }

    /// Set pre-calculated initial temperature
    pub fn with_initial_temperature(mut self, temperature: f64) -> Self {
        self.initial_temperature = Some(temperature);
        self
    }
}

/// State of a single parallel process in ASA
#[derive(Debug, Clone)]
struct ProcessState<S: Clone> {
    solution: S,
    score: f64,
    /// Best solution found by this process
    best_solution: S,
    best_score: f64,
    accepted_moves: usize,
    rejected_moves: usize,
    // Fields for running average and variance calculation (Welford's algorithm)
    cost_count: usize,
    cost_mean: f64,
    cost_m2: f64,  // Sum of squares of differences from the current mean
}

impl<S: Clone> ProcessState<S> {
    fn new(solution: S, score: f64) -> Self {
        Self {
            best_solution: solution.clone(),
            best_score: score,
            solution,
            score,
            accepted_moves: 0,
            rejected_moves: 0,
            cost_count: 0,
            cost_mean: 0.0,
            cost_m2: 0.0,
        }
    }

    fn update_best(&mut self) {
        if self.score < self.best_score {
            self.best_solution = self.solution.clone();
            self.best_score = self.score;
        }
    }

    fn add_cost(&mut self, cost: f64) {
        // Welford's online algorithm for computing variance
        self.cost_count += 1;
        let delta = cost - self.cost_mean;
        self.cost_mean += delta / self.cost_count as f64;
        let delta2 = cost - self.cost_mean;
        self.cost_m2 += delta * delta2;
    }

    fn get_average_cost(&self) -> f64 {
        if self.cost_count == 0 {
            self.score
        } else {
            self.cost_mean
        }
    }

    fn get_variance(&self) -> f64 {
        if self.cost_count < 2 {
            0.0
        } else {
            self.cost_m2 / self.cost_count as f64
        }
    }

    fn reset_stats(&mut self) {
        self.cost_count = 0;
        self.cost_mean = 0.0;
        self.cost_m2 = 0.0;
    }
}

/// Statistics collected during Adaptive SA
#[derive(Debug, Clone)]
pub struct AdaptiveAnnealingStats {
    pub total_steps: usize,
    pub temperature_changes: usize,
    pub initial_temperature: f64,
    pub final_temperature: f64,
    pub initial_score: f64,
    pub final_score: f64,
    pub best_score: f64,
    pub total_accepted_moves: usize,
    pub total_rejected_moves: usize,
    pub initial_omega: f64,
    pub final_omega: f64,
    pub specific_heat_peak_temp: Option<f64>,
    pub score_history: Vec<(usize, f64)>,
}

/// Record for summary data at specific step intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryRecord {
    pub step: usize,
    /// Global best score found so far (across all processes)
    pub best_score: f64,
    /// Average of each process's best solution's basin score
    pub average_best_basin_score: f64,
    /// Average basin score across all processes' current solutions
    pub average_current_basin_score: f64,
    pub current_temperature: f64,
    pub log10_temperature: f64,
    /// Basin of each process's best solution (optional, only recorded at specified steps)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub all_process_best_basins: Option<Vec<Vec<bool>>>,
}

/// Record for specific heat data at each temperature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecificHeatRecord {
    pub temperature: f64,
    pub log10_temperature: f64,
    pub total_steps: usize,
    pub residence_steps: usize,
    pub specific_heat: f64,
}

/// Combined record containing both summary and specific heat data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedRecord {
    pub summary: Vec<SummaryRecord>,
    pub specific_heat: Vec<SpecificHeatRecord>,
}

/// Configuration for experimental recording
#[derive(Debug, Clone)]
pub struct RecordingConfig {
    pub output_dir: String,
    pub trial_index: usize,
    pub enabled: bool,
    /// Steps at which to record all process basins (None = don't record, Some(vec) = record at specified steps)
    pub record_solutions: Option<Vec<usize>>,
}

/// Adaptive Simulated Annealing solver with parallel processes and quasi-equilibrium
pub struct AdaptiveSimulatedAnnealingSolver {
    config: AdaptiveSAConfig,
}

impl AdaptiveSimulatedAnnealingSolver {
    /// Create a new ASA solver with the given configuration
    pub fn new(config: AdaptiveSAConfig) -> Self {
        Self { config }
    }

    /// Create a solver with default configuration
    pub fn default() -> Self {
        Self::new(AdaptiveSAConfig::default())
    }

    /// Solve the optimization problem using Adaptive Simulated Annealing
    pub fn solve<P, S: Clone + 'static, G>(
        &self,
        problem: &dyn OptimizationProblem<P, S, G>,
        initial_solution: S,
        rng: &mut Mt19937GenRand64,
    ) -> (S, AdaptiveAnnealingStats) {
        // Initialize recording if enabled
        let recording_enabled = self.config.recording.is_some();
        let mut summary_records = Vec::new();
        let mut specific_heat_records = Vec::new();
        let mut next_record_step = if recording_enabled { 1 } else { usize::MAX };

        // Initialize parallel processes
        let mut processes = Vec::new();
        let mut best_solution = initial_solution.clone();
        let mut best_score = problem.score(&initial_solution);
        let initial_score = best_score;

        for _ in 0..self.config.num_processes {
            let solution = problem.create_random_solution(rng);
            let score = problem.score(&solution);
            if score < best_score {
                best_solution = solution.clone();
                best_score = score;
            }
            processes.push(ProcessState::new(solution, score));
        }

        // Use pre-calculated initial temperature or calculate from acceptance probability
        let initial_temperature = match self.config.initial_temperature {
            Some(temp) => temp,
            None => self.calculate_initial_temperature(problem, rng),
        };
        let mut current_temperature = initial_temperature;

        // Calculate initial omega (variance)
        let initial_omega = self.calculate_omega(&processes);
        let mut current_omega = initial_omega;

        let mut total_steps = 0;
        let mut temperature_changes = 0;
        let mut total_accepted = 0;
        let mut total_rejected = 0;
        let mut specific_heat_peak_temp = None;

        let mut score_history = Vec::new();
        let record_interval = (self.config.max_steps / 100).max(1);

        while total_steps < self.config.max_steps {
            // Reset cost statistics for new temperature
            for process in &mut processes {
                process.reset_stats();
            }

            // Perform MA steps until quasi-equilibrium with recording
            let equilibrium_steps = self.reach_quasi_equilibrium_with_recording(
                problem,
                &mut processes,
                current_temperature,
                rng,
                &mut total_steps,
                &mut best_solution,
                &mut best_score,
                &mut summary_records,
                &mut next_record_step,
            );

            // Update statistics
            for process in &processes {
                total_accepted += process.accepted_moves;
                total_rejected += process.rejected_moves;
                if process.score < best_score {
                    best_solution = process.solution.clone();
                    best_score = process.score;
                }
            }

            // Calculate specific heat and record it
            let specific_heat = self.calculate_specific_heat(&processes, current_temperature);

            if recording_enabled {
                specific_heat_records.push(SpecificHeatRecord {
                    temperature: current_temperature,
                    log10_temperature: current_temperature.log10(),
                    total_steps,
                    residence_steps: equilibrium_steps,
                    specific_heat,
                });
            }

            // Track peak specific heat temperature
            if specific_heat_peak_temp.is_none() ||
               (specific_heat > 0.1 && current_temperature < initial_temperature * 0.5) {
                specific_heat_peak_temp = Some(current_temperature);
            }

            if total_steps % record_interval == 0 {
                score_history.push((total_steps, best_score));
            }

            // Update temperature
            current_temperature *= self.config.gamma;
            temperature_changes += 1;

            // Optional logging
            if let Some(log_interval) = self.config.log_interval {
                if temperature_changes % log_interval == 0 {
                    current_omega = self.calculate_omega(&processes);
                    println!(
                        "Temperature {}: T={:.6}, Best={:.6}, Omega={:.6}, Steps={}",
                        temperature_changes, current_temperature, best_score, current_omega, total_steps
                    );
                }
            }

            // Early termination if temperature is too low
            if current_temperature < 1e-10 {
                break;
            }
        }

        let final_omega = self.calculate_omega(&processes);
        let final_score = processes.iter().map(|p| p.score).fold(f64::INFINITY, f64::min);

        score_history.push((total_steps, best_score));

        // Save recording files if enabled
        if let Some(recording_config) = &self.config.recording {
            if recording_config.enabled {
                let _ = Self::save_combined_records(
                    recording_config,
                    &summary_records,
                    &specific_heat_records,
                    self.config.initial_acceptance_probability,
                    self.config.epsilon,
                    self.config.gamma,
                    self.config.num_processes,
                );
            }
        }

        let stats = AdaptiveAnnealingStats {
            total_steps,
            temperature_changes,
            initial_temperature,
            final_temperature: current_temperature,
            initial_score,
            final_score,
            best_score,
            total_accepted_moves: total_accepted,
            total_rejected_moves: total_rejected,
            initial_omega,
            final_omega,
            specific_heat_peak_temp,
            score_history,
        };

        (best_solution, stats)
    }

    /// Calculate initial temperature from desired acceptance probability
    /// Runs simulated annealing from 10^2 to 10^-5, recording acceptance rates at each temperature
    /// Repeats 100 times to get accurate measurements, then selects temperature closest to target
    pub fn calculate_initial_temperature<P, S: Clone, G>(
        &self,
        problem: &dyn OptimizationProblem<P, S, G>,
        rng: &mut Mt19937GenRand64,
    ) -> f64 {
        // Temperature range and cooling schedule
        let initial_temp = 100.0;  // 10^2
        let final_temp = 1e-5;     // 10^-5
        let cooling_rate = 0.95;
        let steps_per_temperature = problem.neighbour_size(); // N steps per temperature
        let num_trials = 100; // Number of SA runs for accurate measurement
        
        // Store acceptance rates for each temperature
        let mut temperature_acceptance_rates: Vec<(f64, f64)> = Vec::new();

        // Generate temperature schedule
        let mut temperatures = Vec::new();
        let mut temp = initial_temp;
        while temp > final_temp {
            temperatures.push(temp);
            temp *= cooling_rate;
        }

        let mut current_solutions: Vec<S> = Vec::new();
        let mut current_scores: Vec<f64> = Vec::new();

        for _ in 0..num_trials {
            current_solutions.push(problem.create_random_solution(rng));
            current_scores.push(problem.score(current_solutions.last().unwrap()));
        }

        // For each temperature, measure acceptance rate over multiple trials
        for &test_temperature in &temperatures {
            let mut total_accepted = 0;
            let mut total_rejected = 0;

            // Run multiple trials at this temperature
            for i in 0..num_trials {

                // Run N steps at this temperature
                for _ in 0..steps_per_temperature {
                    let (neighbor, neighbor_score) = problem.random_neighbour(&current_solutions[i], current_scores[i], rng);

                    let accept = if neighbor_score < current_scores[i] {
                        true // Always accept improving moves
                    } else {
                        let delta = current_scores[i] - neighbor_score; // Reversed for minimization
                        let probability = (delta / test_temperature).exp();
                        rng.r#gen::<f64>() < probability
                    };

                    if accept {
                        current_solutions[i] = neighbor;
                        current_scores[i] = neighbor_score;
                        total_accepted += 1;
                    } else {
                        total_rejected += 1;
                    }
                }
            }

            // Calculate average acceptance rate across all trials
            let total_moves = total_accepted + total_rejected;
            if total_moves > 0 {
                let acceptance_rate = total_accepted as f64 / total_moves as f64;
                temperature_acceptance_rates.push((test_temperature, acceptance_rate));
            }
        }

        // Find temperature with acceptance rate closest to target
        let target_ps = self.config.initial_acceptance_probability;
        let mut best_temperature = initial_temp;
        let mut best_diff = f64::INFINITY;

        for &(temp, acceptance_rate) in &temperature_acceptance_rates {
            let diff = (acceptance_rate - target_ps).abs();
            if diff < best_diff {
                best_diff = diff;
                best_temperature = temp;
            }
        }

        // Log the selected temperature and its acceptance rate for debugging
        if self.config.log_interval.is_some() {
            println!("Initial temperature estimation for ps={:.2}:", target_ps);
            println!("  Selected temperature: {:.6}", best_temperature);
            println!("  Achieved acceptance rate: {:.4}",
                     temperature_acceptance_rates.iter()
                         .find(|&&(t, _)| (t - best_temperature).abs() < 1e-10)
                         .map(|&(_, a)| a)
                         .unwrap_or(0.0));
        }

        best_temperature
    }

    /// Perform MA steps until quasi-equilibrium is reached with recording
    fn reach_quasi_equilibrium_with_recording<P, S: Clone + 'static, G>(
        &self,
        problem: &dyn OptimizationProblem<P, S, G>,
        processes: &mut [ProcessState<S>],
        temperature: f64,
        rng: &mut Mt19937GenRand64,
        total_steps: &mut usize,
        best_solution: &mut S,
        best_score: &mut f64,
        summary_records: &mut Vec<SummaryRecord>,
        next_record_step: &mut usize,
    ) -> usize {
        let mut steps = 0;
        let recording_enabled = self.config.recording.is_some();
        let record_solution_steps: Option<&Vec<usize>> = self.config.recording
            .as_ref()
            .and_then(|r| r.record_solutions.as_ref());

        // Calculate initial omega for this temperature
        let initial_omega = self.calculate_omega(processes);

        loop {
            steps += 1;
            *total_steps += 1;

            // Perform one MA step for each process
            for process in processes.iter_mut() {
                let (neighbor, neighbor_score) = problem.random_neighbour(&process.solution, process.score, rng);

                let accept = if neighbor_score < process.score {
                    true
                } else if temperature > 0.0 {
                    let delta = process.score - neighbor_score; // Reversed for minimization
                    let probability = (delta / temperature).exp();
                    rng.r#gen::<f64>() < probability
                } else {
                    false
                };

                if accept {
                    process.solution = neighbor;
                    process.score = neighbor_score;
                    process.accepted_moves += 1;

                    // Update process-level best
                    process.update_best();

                    // Update global best
                    if process.score < *best_score {
                        *best_solution = process.solution.clone();
                        *best_score = process.score;
                    }
                } else {
                    process.rejected_moves += 1;
                }

                // Add cost to history for omega calculation
                process.add_cost(process.score);
            }

            // Check if we need to record at this step
            if recording_enabled && *total_steps >= *next_record_step {
                // Check if we should record basins at this step
                let should_record_basins = record_solution_steps
                    .map_or(false, |steps| steps.contains(total_steps));

                // Calculate basin scores for each process's BEST solution
                let best_basin_scores: Vec<f64> = processes.iter().map(|process| {
                    let basin_solution = problem.basin(&process.best_solution);
                    problem.score(&basin_solution)
                }).collect();
                let average_best_basin_score = best_basin_scores.iter().sum::<f64>() / best_basin_scores.len() as f64;

                // Calculate basin scores for each process's CURRENT solution
                let current_basin_scores: Vec<f64> = processes.iter().map(|process| {
                    let basin_solution = problem.basin(&process.solution);
                    problem.score(&basin_solution)
                }).collect();
                let average_current_basin_score = current_basin_scores.iter().sum::<f64>() / current_basin_scores.len() as f64;

                // Record basins of best solutions if requested
                let all_process_best_basins = if should_record_basins {
                    let best_basins: Vec<Option<Vec<bool>>> = processes.iter().map(|process| {
                        let basin_solution = problem.basin(&process.best_solution);
                        let basin_any: &dyn Any = &basin_solution;
                        basin_any.downcast_ref::<Vec<bool>>().cloned()
                    }).collect();

                    if best_basins.iter().all(|b| b.is_some()) {
                        Some(best_basins.into_iter().map(|b| b.unwrap()).collect())
                    } else {
                        None
                    }
                } else {
                    None
                };

                summary_records.push(SummaryRecord {
                    step: *total_steps,
                    best_score: *best_score,
                    average_best_basin_score,
                    average_current_basin_score,
                    current_temperature: temperature,
                    log10_temperature: temperature.log10(),
                    all_process_best_basins,
                });

                *next_record_step = Self::next_recording_step(*total_steps);
            }

            // Recalculate omega at every step and check quasi-equilibrium condition
            let current_omega = self.calculate_omega(processes);
            let equilibrium_ratio = if initial_omega > 0.0 {
                current_omega / initial_omega
            } else {
                1.0
            };

            if equilibrium_ratio < self.config.epsilon || *total_steps >= self.config.max_steps {
                break;
            }

        }

        steps
    }

    /// Calculate Omega (variance measure) for quasi-equilibrium
    fn calculate_omega<S: Clone>(&self, processes: &[ProcessState<S>]) -> f64 {
        if processes.is_empty() {
            return 0.0;
        }

        let process_averages: Vec<f64> = processes.iter().map(|p| p.get_average_cost()).collect();
        let global_average = process_averages.iter().sum::<f64>() / process_averages.len() as f64;

        let variance = process_averages
            .iter()
            .map(|avg| (avg - global_average).powi(2))
            .sum::<f64>() / processes.len() as f64;

        variance
    }

    /// Calculate specific heat C(T) = (<f²> - <f>²) / T²
    /// Calculates specific heat for each process separately and returns the average
    fn calculate_specific_heat<S: Clone>(&self, processes: &[ProcessState<S>], temperature: f64) -> f64 {
        if temperature <= 0.0 || processes.is_empty() {
            return 0.0;
        }

        let mut specific_heats = Vec::new();

        for process in processes {
            if process.cost_count < 2 {
                continue;
            }

            let variance_f = process.get_variance();
            let process_specific_heat = variance_f / (temperature * temperature);
            specific_heats.push(process_specific_heat);
        }

        if specific_heats.is_empty() {
            return 0.0;
        }

        // Return the average specific heat across all processes
        specific_heats.iter().sum::<f64>() / specific_heats.len() as f64
    }

    /// Check if a step number should be recorded (only powers of 10: 10^3, 10^4, 10^5, ...)
    fn should_record_step(step: usize) -> bool {
        if step < 1000 {
            return false;
        }
        // Check if step is a power of 10
        let log10 = (step as f64).log10();
        (log10 - log10.round()).abs() < 1e-9
    }

    /// Generate the next recording step after the given step (next power of 10, minimum 10^3)
    fn next_recording_step(current_step: usize) -> usize {
        if current_step < 1000 {
            return 1000;
        }
        // Find next power of 10
        let current_log10 = (current_step as f64).log10().floor() as u32;
        10_usize.pow(current_log10 + 1)
    }

    /// Save combined records (summary and specific heat) to JSON file
    fn save_combined_records(
        config: &RecordingConfig,
        summary_records: &[SummaryRecord],
        specific_heat_records: &[SpecificHeatRecord],
        ps: f64,
        epsilon: f64,
        gamma: f64,
        num_processes: usize
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create the directory if it doesn't exist
        fs::create_dir_all(&config.output_dir)?;

        let filename = format!(
            "ps{}_eps{}_gamma{}_I{}_trial{}.json",
            ps, epsilon, gamma, num_processes, config.trial_index
        );
        let filepath = Path::new(&config.output_dir).join(filename);

        // Create combined record
        let combined = CombinedRecord {
            summary: summary_records.to_vec(),
            specific_heat: specific_heat_records.to_vec(),
        };

        let json = serde_json::to_string_pretty(&combined)?;
        fs::write(filepath, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_partition::{Graph, GraphPartitionProblem};
    use rand_mt::Mt19937GenRand64;

    #[test]
    fn test_cooling_schedule_constant() {
        let solver = SimulatedAnnealingSolver::new(
            SimulatedAnnealingConfig::new_constant(50.0, 10)
        );
        
        // Test constant temperature
        assert!((solver.calculate_temperature(0) - 50.0).abs() < 1e-10);
        assert!((solver.calculate_temperature(1) - 50.0).abs() < 1e-10);
        assert!((solver.calculate_temperature(5) - 50.0).abs() < 1e-10);
        assert!((solver.calculate_temperature(100) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = SimulatedAnnealingConfig::new_constant(50.0, 5000)
            .with_logging(100);
        
        let CoolingSchedule::Constant { temperature } = config.cooling_schedule;
        assert!((temperature - 50.0).abs() < 1e-10);
        assert_eq!(config.log_interval, Some(100));
        assert_eq!(config.max_iterations, 5000);
    }

    #[test]
    fn test_solver_with_graph_partition_problem() {
        // Create a test graph
        let mut graph = Graph::new(6);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph.add_edge(4, 5);
        graph.add_edge(5, 0);
        
        let problem = GraphPartitionProblem::new(graph);
        let initial_solution = vec![true, false, true, false, true, false];
        
        // Create solver with constant temperature
        let config = SimulatedAnnealingConfig::new_constant(10.0, 1000);
        let solver = SimulatedAnnealingSolver::new(config);
        
        let mut rng = Mt19937GenRand64::new(42);
        let (solution, stats) = solver.solve(&problem, initial_solution.clone(), &mut rng);
        
        // Verify results
        assert_eq!(solution.len(), 6);
        assert!(stats.iterations_completed > 0);
        assert!(stats.best_score <= stats.initial_score + 1e-10); // Should find same or better (minimization)
        assert!(stats.accepted_moves + stats.rejected_moves > 0);
        assert!((stats.initial_score - problem.score(&initial_solution)).abs() < 1e-10);
    }

    #[test]
    fn test_solver_with_different_temperatures() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 0);
        
        let problem = GraphPartitionProblem::new(graph);
        let initial_solution = vec![true, true, false, false];
        
        // Test with different constant temperatures
        let temperatures = [1.0, 5.0, 10.0];
        
        for (i, &temp) in temperatures.iter().enumerate() {
            let config = SimulatedAnnealingConfig::new_constant(temp, 500);
            let solver = SimulatedAnnealingSolver::new(config);
            let mut rng = Mt19937GenRand64::new(12345 + i as u64);
            let (solution, stats) = solver.solve(&problem, initial_solution.clone(), &mut rng);
            
            assert_eq!(solution.len(), 4);
            assert!(stats.iterations_completed > 0);
            assert!(stats.best_score.is_finite());
            assert!(stats.accepted_moves + stats.rejected_moves > 0);
        }
    }

    #[test]
    fn test_full_iterations() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        
        let problem = GraphPartitionProblem::new(graph);
        let initial_solution = vec![true, false, true];
        
        // Create config that should run all iterations
        let config = SimulatedAnnealingConfig::new_constant(2.0, 100);
        
        let solver = SimulatedAnnealingSolver::new(config);
        let mut rng = Mt19937GenRand64::new(999);
        let (solution, stats) = solver.solve(&problem, initial_solution, &mut rng);
        
        // Should complete all iterations
        assert_eq!(stats.iterations_completed, 100);
        assert!((stats.final_temperature - 2.0).abs() < 1e-10);
        assert_eq!(solution.len(), 3);
    }

    #[test]
    fn test_stats_collection() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(2, 3);
        
        let problem = GraphPartitionProblem::new(graph);
        let initial_solution = vec![true, false, false, true];
        let initial_score = problem.score(&initial_solution);
        
        let config = SimulatedAnnealingConfig::new_constant(2.0, 100);
        let solver = SimulatedAnnealingSolver::new(config);
        
        let mut rng = Mt19937GenRand64::new(777);
        let (solution, stats) = solver.solve(&problem, initial_solution, &mut rng);
        
        // Verify stats are meaningful
        assert_eq!(stats.initial_score, initial_score);
        assert_eq!(stats.final_score, problem.score(&solution));
        assert!(stats.best_score <= stats.final_score + 1e-10); // Best should be <= final (minimization)
        assert!(stats.accepted_moves + stats.rejected_moves > 0);
        assert!(stats.iterations_completed <= 100);
        assert!(stats.final_temperature >= 0.0);
    }
}