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
    * @brief Generate laser pulse envelopes with different temporal shapes (Gaussian, box, sech).
    * Creates piecewise constant signals for Boulder Opal optimization with smooth temporal profiles.
    * Pulse area Theta = ∫ Omega(t) dt determines rotation angle for quantum gate operations.
    *
    * @param graph Boulder Opal graph object for tensor operations.
    * @param pulse_shape Temporal shape: "gaus", "box", or "sech".
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
    * @param Omega1 First laser Rabi frequency in rad/s.
    * @param Omega2 Second laser Rabi frequency in rad/s.
    * @param pulse_shape Temporal pulse shape: "gaus", "box", or "sech".
    * @param width Pulse width in seconds.
    * @param sigma_p Momentum spread parameter (0=no Doppler effects).
    * @param sigma_b Intensity variation parameter (0=uniform beam).
    * @param noise_max Phase noise amplitude in radians (0=perfect coherence).
    * @return Dict with "states" |psi(t)⟩ and "unitaries" U(t) tensors.
    """

# =============================================================================
# COST FUNCTIONS AND OPTIMIZATION
# =============================================================================

def calculate_cost_states_doc():
    """
    * @brief Compute gate infidelity cost function for quantum gate optimization.
    * Cost = (1/N) ∑ |1 - |⟨psi_target|U(tᵢ)|psi_0⟩|²|.
    * Target states: beam-splitter "bs" → (|1⟩ + |3⟩)/√2, mirror "m" → |3⟩.
    *
    * @param graph Boulder Opal graph object for tensor operations.
    * @param pulse_type Gate type: "bs" (beam-splitter) or "m" (mirror).
    * @param unitaries Time evolution operators U(t) from calculate_unitary.
    * @return Boulder Opal tensor - scalar cost value to minimize (0=perfect, 1=worst).
    """

def single_optimisation_doc():
    """
    * @brief Perform pulse optimization using Adam optimizer for single learning rate.
    * Optimizes Rabi frequencies Omega_1/Omega_2, pulse width tau, and time-dependent detuning delta(t).
    * Uses automatic differentiation through quantum evolution with exact gradients.
    *
    * @param learning_rate Adam optimizer learning rate (typical 1e-3 to 1e-1).
    * @param pulse_shape Temporal pulse shape: "gaus", "box", or "sech".
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
    * Generates before/after plots, cost histories, and saves optimization results to .pkl files.
    *
    * @param pulse_shape Temporal pulse shape: "gaus", "box", or "sech" (default "gaus").
    * @param pulse_type Gate type: "bs" (beam-splitter) or "m" (mirror) (default "bs").
    * @param sigma_p Momentum spread parameter for Doppler effects (default 0.0).
    * @param sigma_b Intensity variation parameter for spatial effects (default 0.0).
    * @param noise_max Phase noise amplitude in radians (default 0.0).
    * @param width Optional pulse width override in seconds.
    * @param target Optional target state override for custom gates.
    * @return Optimized pulses with plots and data files saved to organized directory structure.
    """

# =============================================================================
# SYSTEM ANALYSIS AND UTILITIES
# =============================================================================

def preoptimize_pulse_doc():
    """
    * @brief Automatically determine optimal initial pulse width by testing range from initial_width to duration/2.
    * Success criteria: beam-splitter requires equal populations |P₁(t) - P₃(t)| < 0.0005, mirror requires P₁(t) < 0.0005.
    * Uses parallel execution to test multiple widths and returns first achieving target fidelity.
    *
    * @param pulse_shape Temporal pulse shape: "gaus", "box", or "sech".
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

# =============================================================================
# MAIN EXECUTION AND WORKFLOW
# =============================================================================

def main_execution_doc():
    """
    * @brief Orchestrate systematic pulse optimization across multiple noise parameters to characterize system performance.
    * Sequence: setup signal handlers → initialize parameters → systematic sweep (momentum, intensity, phase noise, combined effects) → test all pulse shapes and gate types.
    * Generates organized output with plots and data files, requires 2-8 hours for complete characterization.
    *
    * @param None Automated parameter sweeps with predefined noise levels and pulse combinations.
    * @return Complete analysis results saved to plots/ and data/ directories.
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
   
   Solutions:
   - Try different learning rates: [10⁻⁴, 10⁻³, 10⁻²]
   - Increase nb_iterations (double current value)
   - Check preoptimize_pulse() success
   - Reduce noise parameters for testing
   - Verify physical parameters are reasonable

2. Memory Errors:
   
   Symptoms: Out of memory, system freezing
   Causes:
   - Too many workers for available RAM
   - Large time_count with noise batches
   - Memory leaks from incomplete cleanup
   
   Solutions:
   - Reduce max_workers (try cpu_count // 4)
   - Reduce time_count (try 2000-3000)
   - Reduce momentum_batch/intensity_batch
   - Monitor memory usage during execution
   - Ensure proper cleanup (force_cleanup())

3. Slow Execution:
   
   Symptoms: Very long optimization times
   Causes:
   - Too many workers (contention)
   - Large batch dimensions
   - High time resolution
   - Inefficient parallelization
   
   Solutions:
   - Optimize max_workers (try different values)
   - Profile with smaller problems first
   - Use background execution for long runs
   - Consider reducing time_count for testing

4. Optimization Failures:
   
   Symptoms: NaN costs, optimization exceptions
   Causes:
   - Numerical instability
   - Invalid parameter bounds
   - Boulder Opal graph errors
   
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
    - calculate_evolution
    - calculate_cost_states
    - single_optimisation
    - optimize_pulse
    - preoptimize_pulse
    - plotting
    - estimate_number_workers
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
        "single_optimisation": single_optimisation_doc.__doc__,
        "optimize_pulse": optimize_pulse_doc.__doc__,
        "preoptimize_pulse": preoptimize_pulse_doc.__doc__,
        "plotting": plotting_doc.__doc__,
        "estimate_number_workers": estimate_number_workers_doc.__doc__,
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
