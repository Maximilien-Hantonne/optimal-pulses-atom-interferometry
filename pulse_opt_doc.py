"""
Pulse Optimization Documentation
================================

This module contains comprehensive documentation for all functions and concepts
used in the pulse optimization code for three-level atomic systems.

Author: Maximilien Hantonne
Date: July 2025
"""

# =============================================================================
# PHYSICAL SYSTEM OVERVIEW
# =============================================================================

SYSTEM_OVERVIEW = """
Three-Level Atomic System (Λ-type configuration)
===============================================

The system models Rubidium-87 atoms in a three-level Λ configuration:

States:
-------
|1⟩ = |F=1, mF=0⟩  (Ground state)
|2⟩ = |F=2, mF=0⟩  (Intermediate excited state)  
|3⟩ = |F=1, mF=0⟩  (Final ground state)

Transitions:
-----------
|1⟩ ←--Omega_1--> |2⟩  (First laser, ~780 nm)
|2⟩ ←--Omega_2--> |3⟩  (Second laser, ~780 nm)

Key Parameters:
--------------
- Delta: One-photon detuning (typically ~100 MHz)
- delta: Two-photon detuning (near resonance, ~0-1 kHz)
- Omega_1, Omega_2: Rabi frequencies (typically ~1-10 MHz)
- Duration: Pulse duration (typically 100-1000 mu_s)

Target Operations:
-----------------
- Beam-splitter (bs): |0⟩ → (|0⟩ + |2⟩)/√2
- Mirror (m): |0⟩ → |2⟩
"""

PARAMETER_GUIDELINES = """
Parameter Selection Guidelines
=============================

Pulse Shapes:
------------
- "gaus": Gaussian envelope, smooth, good for general optimization
- "box": Rectangular envelope with smooth edges, fast transitions
- "sech": Hyperbolic secant, optimal for certain quantum operations
- "free": Fully optimizable shape, maximum flexibility but slower convergence

Noise Parameters:
----------------
- sigma_p: Momentum spread (0.01-0.3 typical, 0.0 = no Doppler effects)
- sigma_b: Intensity variation (0.3-3.0 typical, 0.0 = uniform beam)
- noise_max: Phase noise amplitude (1e-6 to 1e-2 rad, 0.0 = perfect coherence)

Optimization Settings:
---------------------
- learning_rate: 1e-4 to 1e-1 (auto-selected from range)
- nb_iterations: 2000-8000 (more for complex optimization)
- time_count: 2000-5000 (higher for better resolution)
- duration: 100e-6 to 1000e-6 seconds

Resource Management:
-------------------
- max_workers: cpu_count()//2 to cpu_count()-2
- Memory usage scales with: time_count × batch_dimensions × max_workers
- Reduce parameters if memory issues occur
"""

# =============================================================================
# NOISE AND DISTRIBUTION FUNCTIONS
# =============================================================================

def momentum_distribution_doc():
    """
    * @brief Create Gaussian momentum distribution to model atomic velocity spread in MOT.
    * Creates Doppler shifts: omega_effective = omega_laser ± k·v where k is laser wavevector.
    * Returns momentum samples scaled by (hbar·k_eff/m_Rb) for energy units.
    *
    * @param graph Boulder Opal graph object for tensor operations.
    * @param sigma Momentum spread width (dimensionless, typical 0.01-0.3).
    * @param batch_dim Tensor batch dimension for parallel processing.
    * @return Tuple of (momentum_signal, updated_batch_dim).
    """

def intensity_distribution_doc():
    """
    * @brief Model spatial variation in laser intensity across atomic cloud with Gaussian profile.
    * Atoms at different positions experience different Rabi frequencies: Omega(r) = Omega_0 · √(I(r)/I₀).
    * Returns intensity samples affecting effective Rabi frequency Omega_eff(x) = Omega_0 · √(beta(x)).
    *
    * @param graph Boulder Opal graph object for tensor operations.
    * @param sigma Intensity profile width (dimensionless, typical 0.3-3.0).
    * @param batch_dim Tensor batch dimension for parallel processing.
    * @param beta_min Minimum intensity fraction (default 0.0).
    * @param beta_max Maximum intensity fraction (default 1.0).
    * @return Tuple of (intensity_signal, updated_batch_dim).
    """

def phase_noise_doc():
    """
    * @brief Generate realistic laser phase noise with 1/f^alpha power spectral density.
    * Modifies laser field as E(t) = E₀ · exp(i[omega*t + phi(t)]) causing dephasing and reduced fidelity.
    * Uses colored noise generation with frequency cutoff to prevent unphysical high-frequency components.
    *
    * @param graph Boulder Opal graph object for tensor operations.
    * @param noise_max Maximum RMS amplitude in radians (typical 1e-6 to 1e-2).
    * @param alpha Power law exponent for 1/f^alpha noise (default 1).
    * @param cutoff Frequency cutoff in Hz (default 5e4).
    * @return Boulder Opal pwc_signal with time-dependent phase noise phi(t).
    """

# =============================================================================
# PULSE GENERATION FUNCTIONS
# =============================================================================

def pulse_doc():
    """
    * @brief Generate laser pulse envelopes with different temporal shapes (Gaussian, box, sech, free).
    * Creates piecewise constant signals for Boulder Opal optimization with smooth temporal profiles.
    * Pulse area Theta = ∫ Omega(t) dt determines rotation angle for quantum gate operations.
    * Free shape uses real_optimizable_pwc_signal for completely flexible pulse optimization.
    *
    * @param graph Boulder Opal graph object for tensor operations.
    * @param pulse_shape Temporal shape: "gaus", "box", "sech", or "free".
    * @param width Pulse duration in seconds (typical 10-100 microseconds).
    * @param amplitude Peak Rabi frequency in rad/s (typical 2*pi*1MHz).
    * @param name Signal name identifier (default "pulse").
    * @return Boulder Opal pwc_signal with piecewise constant pulse envelope.
    """

def const_pwc_doc():
    """
    * @brief Create constant piecewise-constant signal over full pulse duration.
    * Used for constant detunings (Delta, delta), DC offsets, and reference signals.
    *
    * @param graph Boulder Opal graph object for tensor operations.
    * @param value Constant value throughout duration (dimensionless or in rad/s).
    * @return Boulder Opal pwc_signal with time_count segments of constant value.
    """

# =============================================================================
# HAMILTONIAN AND EVOLUTION FUNCTIONS
# =============================================================================

def set_hamiltonian_doc():
    """
    * @brief Construct complete three-level atomic Hamiltonian H = H₀ + H_kinetic + H_laser.
    * Includes energy levels, kinetic shifts from atomic motion, and laser coupling with noise.
    * Uses matrix representation in {|1⟩, |2⟩, |3⟩} basis with all terms made Hermitian.
    *
    * @param graph Boulder Opal graph object for tensor operations.
    * @param Delta One-photon detuning as pwc_signal (typical 2*pi*100MHz).
    * @param delta Two-photon detuning as pwc_signal (typical 2*pi*1kHz).
    * @param pulse1 First laser Rabi frequency Omega_1(t) as pwc_signal.
    * @param pulse2 Second laser Rabi frequency Omega_2(t) as pwc_signal.
    * @param momentum Momentum distribution for Doppler shifts.
    * @param betas Intensity distribution for spatial variations.
    * @param random_phi_1 Phase noise for first laser.
    * @param random_phi_2 Phase noise for second laser.
    * @return Boulder Opal hermitian_tensor - complete system Hamiltonian H(t).
    """

def calculate_unitary_doc():
    """
    * @brief Compute time evolution operators U(t) by solving Schrödinger equation.
    * Solves i*hbar * ∂U/∂t = H(t) U(t) for piecewise-constant Hamiltonians.
    * For piecewise-constant H: U(t) = ∏ᵢ exp(-i*H_i·Delta_t_i/hbar).
    *
    * @param graph Boulder Opal graph object for tensor operations.
    * @param hamiltonian Time-dependent Hamiltonian H(t) as hermitian_tensor.
    * @return Boulder Opal tensor - unitary operators U(tᵢ) with shape (time_count, [batch_dims], 3, 3).
    """

def calculate_evolution_doc():
    """
    * @brief Master function performing complete quantum simulation of three-level system with realistic noise.
    * Pipeline: generate pulses → create noise distributions → build Hamiltonian → solve evolution → return states/unitaries.
    * Handles multiple batch dimensions for momentum/intensity samples and preserves probability conservation.
    *
    * @param graph Boulder Opal graph object for tensor operations.
    * @param Delta_signal One-photon detuning profile as pwc_signal.
    * @param delta_signal Two-photon detuning profile as pwc_signal.
    * @param pulse1 First laser pulse (None generates automatically based on pulse_shape).
    * @param pulse2 Second laser pulse (None generates automatically based on pulse_shape).
    * @param sigma_p Momentum spread parameter (0=no Doppler effects).
    * @param sigma_b Intensity variation parameter (0=uniform beam).
    * @param noise_max Phase noise amplitude in radians (0=perfect coherence).
    * @param output_node_names List of Boulder Opal output nodes to compute.
    * @return Dict with "states" |psi(t)⟩ and "unitaries" U(t) tensors.
    """

# =============================================================================
# COST FUNCTIONS AND OPTIMIZATION
# =============================================================================

def calculate_cost_states_doc():
    """
    * @brief Compute gate infidelity cost function for quantum gate optimization using state fidelity.
    * Cost = (1/N) ∑ |1 - |⟨psi_target|U(tᵢ)|psi_0⟩|²|.
    * Target states: beam-splitter "bs" → (|1⟩ + |3⟩)/√2, mirror "m" → |3⟩.
    * Averages fidelity over second half of time evolution for steady-state evaluation.
    *
    * @param graph Boulder Opal graph object for tensor operations.
    * @param pulse_type Gate type: "bs" (beam-splitter) or "m" (mirror).
    * @param unitaries Time evolution operators U(t) from calculate_unitary.
    * @return Boulder Opal tensor - scalar cost value to minimize (0=perfect, 1=worst).
    """

def calculate_cost_unitaries_doc():
    """
    * @brief Compute gate infidelity cost function using unitary fidelity at specific time.
    * Cost = |1 - |Tr(U_target† @ U_achieved)|²/Tr(U_target† @ U_target)|².
    * Alternative to state-based cost function, more direct for gate characterization.
    * Used when cost_type="unitaries" in optimization settings.
    *
    * @param graph Boulder Opal graph object for tensor operations.
    * @param unitaries Achieved unitary operator U from quantum evolution.
    * @param target_unitaries Target unitary operator for gate operation.
    * @return Boulder Opal tensor - scalar cost value to minimize (0=perfect, 1=worst).
    """

def single_optimisation_doc():
    """
    * @brief Perform pulse optimization using Adam optimizer for single learning rate.
    * For standard shapes (gaus, box, sech): optimizes Rabi frequencies Omega_1/Omega_2, pulse width tau, and time-dependent detuning delta(t).
    * For free shape: optimizes pulse envelopes directly and time-dependent detuning delta(t) with fixed amplitudes.
    * Uses automatic differentiation through quantum evolution with exact gradients.
    *
    * @param learning_rate Adam optimizer learning rate (typical 1e-3 to 1e-1).
    * @param target_unitary Target unitary operator for gate fidelity calculation.
    * @param target_index Time index for unitary evaluation in cost function.
    * @param pulse_shape Temporal pulse shape: "gaus", "box", "sech", or "free".
    * @param pulse_type Gate type: "bs" (beam-splitter) or "m" (mirror).
    * @param sigma_p Momentum spread parameter for Doppler effects.
    * @param sigma_b Intensity variation parameter for spatial effects.
    * @param noise_max Phase noise amplitude in radians.
    * @param width Initial pulse width in seconds.
    * @param nb_iter Number of optimization iterations.
    * @return Tuple of (learning_rate, optimization_result) with optimized parameters and cost history.
    """

def optimize_pulse_doc():
    """
    * @brief Master optimization function with two-stage strategy: learning rate survey then extended optimization.
    * Tests multiple learning rates in parallel, selects best, then runs fine-tuning with 2*nb_iterations.
    * Handles both parametric shapes (gaus, box, sech) and free-form optimization.
    * Generates before/after plots, cost histories, and saves optimization results to .pkl files.
    *
    * @param pulse_shape Temporal pulse shape: "gaus", "box", "sech", or "free" (default "gaus").
    * @param pulse_type Gate type: "bs" (beam-splitter) or "m" (mirror) (default "bs").
    * @param sigma_p Momentum spread parameter for Doppler effects (default 0.0).
    * @param sigma_b Intensity variation parameter for spatial effects (default 0.0).
    * @param noise_max Phase noise amplitude in radians (default 0.0).
    * @param width Optional pulse width override in seconds.
    * @param target_unitary Optional target unitary override for custom gates.
    * @param target_index Optional target time index override.
    * @return Optimized pulses with plots and data files saved to organized directory structure.
    """

# =============================================================================
# SYSTEM ANALYSIS AND UTILITIES
# =============================================================================

def preoptimize_pulse_doc():
    """
    * @brief Automatically determine optimal initial pulse width by testing range from initial_width to duration/2.
    * Success criteria: beam-splitter requires equal populations |P₁(t) - P₃(t)| < 0.0005, mirror requires P₁(t) < 0.0005.
    * For free shape, uses Gaussian pulses for initial width determination, then switches to optimizable shape.
    * Uses parallel execution to test multiple widths and returns first achieving target fidelity.
    *
    * @param pulse_shape Temporal pulse shape: "gaus", "box", "sech", or "free".
    * @param pulse_type Gate type: "bs" (beam-splitter) or "m" (mirror).
    * @param initial_width Starting width for search in seconds (default 10e-6).
    * @return Tuple of (optimal_width, target_unitary, target_index) or (None, None, None) if no success.
    """

def plotting_doc():
    """
    * @brief Generate publication-quality plots for quantum pulse optimization results.
    * "states" shows population dynamics P₁(t), P₂(t), P₃(t), "cost" shows optimization convergence, "controls" shows optimized pulse shapes Omega_1(t), Omega_2(t).
    * Creates dual save locations organized by pulse shape and parameters for easy comparison.
    *
    * @param result Optimization result data containing states, costs, and control parameters.
    * @param sigma_p Momentum spread parameter (default 0.0).
    * @param sigma_b Intensity variation parameter (default 0.0).
    * @param noise_max Phase noise amplitude (default 0.0).
    * @param pulse_shape Temporal pulse shape: "gaus", "box", or "sech" (default "gaus").
    * @param pulse_type Gate type: "bs" or "m" (default "bs").
    * @param what Plot type: "states", "cost", or "controls" (default "states").
    * @param when Timing: "before" or "after" optimization (default "before").
    * @return Saves PNG files to organized directory structure with qctrl styling.
    """

def estimate_number_workers_doc():
    """
    * @brief Determine optimal number of parallel workers based on system resources and problem complexity.
    * Currently uses cpu_count() // 2 baseline with memory availability check.
    * Should consider memory per worker, CPU topology, and problem scaling for future enhancements.
    *
    * @param time_count Time resolution parameter affecting memory usage per worker.
    * @param nb_iterations Number of optimization steps affecting computation time.
    * @return int - optimal number of parallel workers for current system.
    """

def initialize_optimization_doc():
    """
    * @brief Initialize global optimization parameters including worker count and learning rate grid.
    * Sets up logarithmic learning rate range for parallel testing across available workers.
    * Called once at start of main execution to configure multiprocessing optimization.
    *
    * @param lr_log_min Minimum learning rate exponent (e.g., -3 for 1e-3).
    * @param lr_log_max Maximum learning rate exponent (e.g., 0 for 1e0).
    * @return None - sets global max_workers and learning_rates arrays.
    """

def force_cleanup_doc():
    """
    * @brief Aggressively clean up system resources to prevent memory leaks and zombie processes.
    * Closes matplotlib figures, forces garbage collection, terminates multiprocessing children.
    * Used after optimization runs and in signal handlers for graceful shutdown.
    *
    * @param None
    * @return None - performs cleanup operations with exception handling.
    """

def run_all_pulses_doc():
    """
    * @brief Execute complete pulse optimization for all shapes (gaus, box, sech, free) and gate types (bs, m).
    * Systematic execution of 8 optimizations total with progress tracking and cleanup.
    * Used for batch parameter sweeps and comprehensive characterization studies.
    *
    * @param sigma_p Momentum spread parameter for all optimizations.
    * @param sigma_b Intensity variation parameter for all optimizations.
    * @param noise_max Phase noise amplitude for all optimizations.
    * @return None - optimized results saved to files, plots generated.
    """

def evaluate_width_doc():
    """
    * @brief Test specific pulse width for achieving target gate fidelity in ideal conditions.
    * Used by preoptimize_pulse() to find initial width that satisfies success criteria.
    * Success criteria: beam-splitter |P₁(end) - P₃(end)| < 0.0005, mirror P₁(end) < 0.0005.
    * For free shape, uses Gaussian pulses for evaluation to maintain consistency.
    *
    * @param width Pulse width to test in seconds.
    * @param pulse_shape Temporal pulse shape: "gaus", "box", "sech", or "free".
    * @param pulse_type Gate type: "bs" (beam-splitter) or "m" (mirror).
    * @return Tuple of (width, end_time) if successful, (None, None) if failed.
    """

# =============================================================================
# MAIN EXECUTION AND WORKFLOW
# =============================================================================

def main_execution_doc():
    """
    * @brief Orchestrate systematic pulse optimization across multiple noise parameters to characterize system performance.
    * Sequence: setup signal handlers → initialize parameters → systematic sweep (momentum, intensity, phase noise, combined effects) → test all pulse shapes and gate types.
    * Parameter sweep includes: momentum (0.01, 0.1, 0.3), intensity (0.3, 1, 3), phase noise (1e-6, 1e-4, 1e-2), and combinations.
    * For each parameter set, optimizes all 4 pulse shapes (gaus, box, sech, free) and 2 gate types (bs, m) = 8 optimizations per parameter set.
    * Generates organized output with plots and data files, requires 2-8 hours for complete characterization.
    *
    * @param None Automated parameter sweeps with predefined noise levels and pulse combinations.
    * @return Complete analysis results saved to plots/ and data/ directories with systematic organization.
    """

# =============================================================================
# TROUBLESHOOTING AND COMMON ISSUES
# =============================================================================

TROUBLESHOOTING_GUIDE = """
Common Issues and Solutions
==========================

1. Poor Convergence (High Final Cost):
   
   Symptoms: Cost plateaus at high values (> 0.1)
   Causes: 
   - Learning rate too high (instability)
   - Learning rate too low (slow convergence)
   - Insufficient iterations
   - Poor initial pulse width
   - Excessive noise
   - Free shape requires more iterations
   
   Solutions:
   - Try different learning rates: [10⁻⁴, 10⁻³, 10⁻²]
   - Increase nb_iterations (double current value)
   - For free shape: use 4000-8000 iterations
   - Check preoptimize_pulse() success
   - Reduce noise parameters for testing
   - Verify physical parameters are reasonable

2. Free Shape Optimization Issues:
   
   Symptoms: Free shape converges poorly compared to parametric shapes
   Causes:
   - Higher dimensional optimization space
   - Insufficient iterations
   - Poor amplitude scaling
   
   Solutions:
   - Use longer optimization (2× iterations)
   - Start with successful parametric optimization
   - Adjust amplitude bounds (Omega_1 * 10)
   - Monitor pulse envelope evolution

3. Memory Errors:
   
   Symptoms: Out of memory, system freezing
   Causes:
   - Too many workers for available RAM
   - Large time_count with noise batches
   - Memory leaks from incomplete cleanup
   - Free shape uses more memory
   
   Solutions:
   - Reduce max_workers (try cpu_count // 4)
   - Reduce time_count (try 2000-3000)
   - Reduce momentum_batch/intensity_batch
   - Monitor memory usage during execution
   - Ensure proper cleanup (force_cleanup())

4. Slow Execution:
   
   Symptoms: Very long optimization times
   Causes:
   - Too many workers (contention)
   - Large batch dimensions
   - High time resolution
   - Inefficient parallelization
   - Free shape optimization complexity
   
   Solutions:
   - Optimize max_workers (try different values)
   - Profile with smaller problems first
   - Use background execution for long runs
   - Consider reducing time_count for testing
   - Start with parametric shapes for testing

5. Optimization Failures:
   
   Symptoms: NaN costs, optimization exceptions
   Causes:
   - Numerical instability
   - Invalid parameter bounds
   - Boulder Opal graph errors
   - Division by zero in cost calculation
   
   Solutions:
   - Check parameter bounds and initial values
   - Validate input signals and noise parameters
   - Reduce learning rates if NaN appears
   - Check for empty arrays in percentile calculations

"""

# =============================================================================
# DOCUMENTATION ACCESS FUNCTIONS
# =============================================================================

def get_function_help(function_name):
    """
    Get detailed documentation for a specific function.
    
    Usage:
    ------
    from pulse_opt_doc import get_function_help
    help_text = get_function_help("momentum_distribution")
    print(help_text)
    
    Available functions:
    -------------------
    - momentum_distribution
    - intensity_distribution  
    - phase_noise
    - pulse
    - const_pwc
    - set_hamiltonian
    - calculate_unitary
    - calculate_evolution
    - calculate_cost_states
    - calculate_cost_unitaries
    - evaluate_width
    - preoptimize_pulse
    - single_optimisation
    - optimize_pulse
    - plotting
    - estimate_number_workers
    - initialize_optimization
    - force_cleanup
    - run_all_pulses
    - main_execution
    """
    
    docs = {
        "momentum_distribution": momentum_distribution_doc.__doc__,
        "intensity_distribution": intensity_distribution_doc.__doc__,
        "phase_noise": phase_noise_doc.__doc__,
        "pulse": pulse_doc.__doc__,
        "const_pwc": const_pwc_doc.__doc__,
        "set_hamiltonian": set_hamiltonian_doc.__doc__,
        "calculate_unitary": calculate_unitary_doc.__doc__,
        "calculate_evolution": calculate_evolution_doc.__doc__,
        "calculate_cost_states": calculate_cost_states_doc.__doc__,
        "calculate_cost_unitaries": calculate_cost_unitaries_doc.__doc__,
        "evaluate_width": evaluate_width_doc.__doc__,
        "preoptimize_pulse": preoptimize_pulse_doc.__doc__,
        "single_optimisation": single_optimisation_doc.__doc__,
        "optimize_pulse": optimize_pulse_doc.__doc__,
        "plotting": plotting_doc.__doc__,
        "estimate_number_workers": estimate_number_workers_doc.__doc__,
        "initialize_optimization": initialize_optimization_doc.__doc__,
        "force_cleanup": force_cleanup_doc.__doc__,
        "run_all_pulses": run_all_pulses_doc.__doc__,
        "main_execution": main_execution_doc.__doc__,
    }
    
    if function_name in docs:
        return docs[function_name]
    else:
        available = ", ".join(docs.keys())
        return f"Function '{function_name}' not found. Available: {available}"

def print_system_overview():
    """Print overview of the physical system."""
    print(SYSTEM_OVERVIEW)

def print_parameter_guidelines():
    """Print parameter selection guidelines."""
    print(PARAMETER_GUIDELINES)

def print_troubleshooting():
    """Print troubleshooting guide."""
    print(TROUBLESHOOTING_GUIDE)

if __name__ == "__main__":
    print("Pulse Optimization Documentation")
    print("="*50)
    print("\nAvailable documentation functions:")
    print("- get_function_help(function_name)")
    print("- print_system_overview()")
    print("- print_parameter_guidelines()")
    print("- print_troubleshooting()")
    print("\nExample usage:")
    print(">>> from pulse_opt8_documentation import get_function_help")
    print(">>> print(get_function_help('calculate_evolution'))")
