import os
import gc
import sys
import time
import pickle
import signal
import psutil
import traceback
import matplotlib
import numpy as np
import multiprocessing as mp

# Avoid shitty warnings
def mute_output():
    sys._stdout = sys.stdout
    sys._stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
def unmute_output():
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = sys._stdout
    sys.stderr = sys._stderr

mute_output()
import boulderopal as bo
import colorednoise as cn
import qctrlvisualizer as qv
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import fsolve
from multiprocessing import cpu_count, set_start_method
from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed

matplotlib.use('Agg') 
plt.style.use(qv.get_qctrl_style())
set_start_method("spawn", force=True)


### AUTHENTICATION

# Authenticate (cloud)
# bo.cloud.set_verbosity("QUIET")
# add you key here

# Authenticate (local)
bo.set_local_mode()
unmute_output()

### PHYSICAL CONSTANTS

# Universal constants
c = 299792458  
epsilon_0 = 8.854187817e-12 
hbar = 1.054571817e-34  
k_b = 1.380649e-23 

# Atomic parameters for Rubidium-87
m_Rb = 86.9092 * 1.6605e-27  
omega_0 = 2 * np.pi * c / 780.241209e-9  # atomic transition frequency 
omega_hf = 2 * np.pi * 6.834682610904e9  # hyperfine splitting
d = 3.58e-29  # dipolar moment
p_0 = 0.0 # initial momentum

##### PARAMETERS

Delta = 2 * np.pi * 1.0e8 # 1-photon detuning
delta =  2 * np.pi * 0.0  # 2-photon detuning
delta_max = 2 * np.pi * 1.0e3 # Maximum value of the 2-photon detuning
Omega_1 = 2 * np.pi * 2.0e6 # Rabi frequency for the first transition
Omega_2 = 2 * np.pi * 2.0e6 # Rabi frequency for the second transition
phi_1 = 2 * np.pi * 0.0 # Phase for the first laser
phi_2 = 2 * np.pi * 0.0  # Phase for the second laser
k_eff = 2 * np.pi / 780e-9  # Effective wavevector

### TIME PARAMETERS

time_count = 5000 # Number of time samples
duration = 700e-6 # Total duration of the simulation
center_time = duration / 2
sample_times = np.linspace(0, duration, time_count) 

### OPIMIZATION PARAMETERS

max_workers = None   # Number of workers for parallel processing
learning_rates = None # Learning rates for optimization
nb_iterations = 4000 # Number of iterations for optimization
cost_type = "states" # Type of cost function to use
# cost_type = "unitaries" # Type of cost function to use

### BATCH DIMENSIONS

momentum_batch = max(time_count//1000, 50)  # Number of momentum samples
intensity_batch = max(time_count//1000, 50)  # Number of intensity samples

### LASER PARAMETERS

# Useless for optimization just to know what the parameters mean in term of wavelength and intensity
def laser_parameters(Delta, delta, Omega1, Omega2):
    def equations(vars):
        lambda1, lambda2 = vars
        k1, k2 = 2 * np.pi / lambda1, 2 * np.pi / lambda2
        omega1, omega2 = 2 * np.pi * c / lambda1, 2 * np.pi * c / lambda2
        keff = k1 + k2
        Delta_calc = omega1 - omega_0 - ((p_0 + hbar * k1)**2 - p_0**2) / (2 * m_Rb * hbar)
        delta_calc = omega1 - omega2 - omega_hf - ((p_0 + hbar * keff)**2 - p_0**2) / (2 * m_Rb * hbar)
        return [Delta_calc - Delta, delta_calc - delta]
    lambda1, lambda2 = fsolve(equations, [780e-9, 780e-9])
    omega1, omega2 = 2 * np.pi * c / lambda1, 2 * np.pi * c / lambda2
    I1 = ((hbar * Omega1) / d)**2 * c * epsilon_0 / 2
    I2 = ((hbar * Omega2) / d)**2 * c * epsilon_0 / 2
    print(f"lambda_1 = {lambda1*1e9:.6f} mathrm{{nm}}, lambda_2 = {lambda2*1e9:.6f} nm$")
    print(f"omega_1 = {omega1:.3e}\\ \\mathrm{{rad/s}},\\ \\omega_2 = {omega2:.3e} rad/s$")
    print(f"I_1 = {I1:.2e} W/m^2, I_2 = {I2:.2e}\\ W/m^2$")

### MATRICES

identity1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
identity2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
identity3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
sigma_m12 = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
sigma_m23 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])

### MOMENTUM DISTRIBUTION

# Return a momentum distribution based on a Gaussian distribution
def momentum_distribution(graph, sigma, batch_dim):

    # If sigma is zero, return a constant zero momentum signal
    if sigma <= 0.0:
        return graph.constant_pwc(constant=0.0, duration=duration), batch_dim
    
    # If sigma is positive, generate random momentum samples with deviation sigma
    momentum_samples = graph.random.normal(
        standard_deviation=sigma,
        mean=p_0,
        shape=(momentum_batch,),
        seed=None,)
    momentum = graph.constant_pwc(
        constant=momentum_samples[:, None, None],
        duration=duration,
        batch_dimension_count=batch_dim,)
    
    return momentum, batch_dim + 1

### INTENSITY DISTRIBUTION

# Return an intensity distribution based on a Gaussian profile
def intensity_distribution(graph, sigma, batch_dim, beta_min=0.0, beta_max=1.0):

    # If sigma is zero, return a constant intensity signal equal to beta_max
    if sigma <= 0.0:
        return graph.constant_pwc(constant=beta_max, duration=duration), batch_dim
    
    # If sigma is positive, generate a Gaussian profile for intensity
    x = np.linspace(0, 1, intensity_batch)
    profile = np.exp(- (x**2) / (2 * sigma**2))
    profile = profile / profile.max()          
    profile = beta_min + (beta_max - beta_min) * profile
    intensity = graph.constant_pwc(
        constant=profile[:, None, None],
        duration=duration,
        batch_dimension_count=batch_dim,)
    
    return intensity, batch_dim + 1

### PHASE NOISE

# Return a colored phase noise signal
def phase_noise(graph, noise_max, alpha=1, cutoff=5e4):

    # If noise_max is zero, return a constant zero phase noise signal
    if noise_max <= 0.0:
        return graph.constant_pwc(constant=0.0, duration=duration)
    
    # If noise_max is positive, generate a colored noise signal with decay equal to alpha
    dt = duration / time_count
    colored_noise = cn.powerlaw_psd_gaussian(exponent=alpha, size=time_count)
    freqs = np.fft.rfftfreq(time_count, d=dt)
    spectrum = np.fft.rfft(colored_noise)
    scaling = np.ones_like(freqs)
    nonzero_freqs = freqs > 0
    scaling[nonzero_freqs] = np.where(
        freqs[nonzero_freqs] < cutoff,
        1.0,
        (cutoff / freqs[nonzero_freqs]) ** alpha)
    scaling[0] = 0.0
    spectrum *= scaling
    hybrid_noise = np.fft.irfft(spectrum, n=time_count)
    hybrid_noise *= noise_max / np.std(hybrid_noise)

    return graph.pwc_signal(duration=duration, values=hybrid_noise)

### PULSE 

# Return a pulse signal with a specified shape and width.
def pulse(graph, pulse_shape, width, amplitude, name="pulse"):

    # Gaussian pulse
    if pulse_shape == "gaus":
        return graph.signals.gaussian_pulse_pwc(
            duration=duration,
            segment_count=time_count,
            amplitude=amplitude,
            width=width,
            center_time=center_time,
            name = name)
    
    # Box pulse
    elif pulse_shape == "box":
        ramp_up = graph.signals.tanh_ramp_pwc(
            duration=duration,
            segment_count=time_count,
            start_value=-amplitude/2,
            end_value=amplitude/2,
            center_time=center_time - width / 2,
            ramp_duration=duration / time_count)
        ramp_down = graph.signals.tanh_ramp_pwc(
            duration=duration,
            segment_count=time_count,
            start_value=amplitude/2,
            end_value=-amplitude/2,
            center_time=center_time + width / 2,
            ramp_duration=duration / time_count)
        total_signal = ramp_up + ramp_down
        total_signal.name = name
        return total_signal
    
    # Sech pulse
    elif pulse_shape == "sech":
        return graph.signals.sech_pulse_pwc(
            duration=duration,
            segment_count=time_count,
            amplitude=amplitude,
            width=width,
            center_time=center_time,
            name= name)
    
    else:
        raise ValueError("Shape not implemented. Choose 'gaus', 'box' or 'sech'.")

### SIGNALS

# Return a constant piecewise constant signal equal to value
def const_pwc(graph, value):
    return graph.constant_pwc(constant=value, duration=duration)

### HAMILTONIAN

# Return the three-level Rabi Hamiltonian
def set_hamiltonian(graph, Delta, delta, pulse1, pulse2, 
                    momentum, betas, random_phi_1, random_phi_2):
    
    # Convert momentum to Doppler shift frequency
    momentum = momentum  * hbar * k_eff**2 / m_Rb

    return (
        graph.hermitian_part(delta * identity1)
        + graph.hermitian_part(-momentum * identity1)
        + graph.hermitian_part((delta + 2 * Delta) * identity2)
        + graph.hermitian_part(-delta * identity3)
        + graph.hermitian_part(momentum * identity3)
        + graph.hermitian_part(betas * pulse1 * sigma_m12 * 
                               graph.exp(phi_1 + random_phi_1))
        + graph.hermitian_part(betas * pulse2 * sigma_m23 * 
                               graph.exp(phi_2 + random_phi_2))
    )

### UNITARY EVOLUTION

# Calculate the unitary evolution operator for the Hamiltonian
def calculate_unitary(graph, hamiltonian):
    return graph.time_evolution_operators_pwc(hamiltonian=hamiltonian, 
                                              sample_times=sample_times,
                                              name="unitaries")

### TIME EVOLUTION

# Calculate the time evolution to return the states and unitaries.
def calculate_evolution(graph, Delta_signal, delta_signal, Omega1, Omega2, 
                        pulse_shape = "gaus",  width = 0.1, sigma_p = 0.0, 
                        sigma_b = 0.0,  noise_max = 0.0,
                        output_node_names=["states", "unitaries"]):
    
    # Create laser pulses
    pulse1 = pulse(graph, pulse_shape, width, Omega1, name="pulse1")
    pulse2 = pulse(graph, pulse_shape, width, Omega2, name="pulse2")

    # Generate momentum and intensity distributions
    batch_dim = 1
    momentum, batch_dim = momentum_distribution(graph, sigma=sigma_p, batch_dim=batch_dim)
    betas, batch_dim = intensity_distribution(graph, sigma=sigma_b, batch_dim=batch_dim)

    # Generate phase noise signals
    random_phi_1 = phase_noise(graph, noise_max=noise_max)
    random_phi_2 = phase_noise(graph, noise_max=noise_max)

    # Simulate the evolution of the system
    hamiltonian = set_hamiltonian(graph, Delta_signal ,delta_signal
                                  ,pulse1, pulse2, momentum, betas, 
                                  random_phi_1, random_phi_2)
    unitaries = calculate_unitary(graph, hamiltonian)
    state = graph.fock_state(3, 0)[:, None]
    states = unitaries @ state
    states.name = "states"
    result = bo.execute_graph(graph=graph, output_node_names=output_node_names)

    return result

### COST

# Calculate the cost for unitaries for gate infidelity
def calculate_cost_unitaries(graph, unitaries, target_unitaries):

    # Calculate the gate infidelity cost for unitaries
    cost = graph.sum(graph.abs(
        1 - graph.abs(graph.trace(graph.adjoint(target_unitaries)  @ unitaries) /
                    graph.trace(graph.adjoint(target_unitaries)  @ target_unitaries))**2))
    
    cost.name = "cost"
    return cost

# Calculate the cost for states for gate infidelity
def calculate_cost_states(graph, pulse_type, unitaries):
    cost = 0.0
    target_index = time_count // 2 + 1

    # Set the target state based on the pulse type
    if pulse_type == "bs":
        target_state = (1 / graph.sqrt(2)) * (
            graph.fock_state(3, 0)[:,None] + graph.fock_state(3, 2)[:,None])
    elif pulse_type == "m":
        target_state = graph.fock_state(3, 2)[:,None]

    # Calculate the gate infidelity cost for states
    for i in range(target_index, time_count):
        cost += graph.abs(1 - graph.abs(graph.trace(graph.adjoint(target_state) @ 
                             unitaries[i,:,:] @ graph.fock_state(3, 0)[:,None]) /
                            graph.trace(graph.adjoint(target_state) @ 
                            target_state)) ** 2) / (time_count - target_index)
        
    cost.name = "cost"
    return cost

### PLOTTING

# Main function for plotting whatever is needed
def plotting(result, sigma_p=0.0, sigma_b=0.0, noise_max=0.0,
             pulse_shape="gaus", pulse_type="bs", what="states", when="before"):
    
    # Create directories for saving plots
    param_parts = []
    if sigma_p != 0.0:
        param_parts.append(f"p_{sigma_p}")
    if sigma_b != 0.0:
        param_parts.append(f"b_{sigma_b}")
    if noise_max != 0.0:
        param_parts.append(f"n_{noise_max}")
    param_key = "_".join(param_parts) if param_parts else "perfect"
    shape_dir = "plots/sorted_plots"
    param_dir = "plots/parameters"
    shape_path = os.path.join(shape_dir, pulse_type, pulse_shape, param_key)
    os.makedirs(shape_path, exist_ok=True)
    param_path = os.path.join(param_dir, param_key)
    os.makedirs(param_path, exist_ok=True)
    filename_parts = [pulse_type, pulse_shape]
    filename_parts.extend(param_parts)

    # Plotting the dynamics of the population
    if what == "states" :
        populations = np.abs(result["output"]["states"]["value"].squeeze())**2
        for non_null in [sigma_p != 0.0, sigma_b != 0.0]:
            if non_null:
                populations = np.sum(populations, axis=0)
        populations = populations / ((momentum_batch if sigma_p else 1) *
                                      (intensity_batch if sigma_b else 1))
        fig = plt.figure()
        qv.plot_population_dynamics(
            sample_times, {rf"$|{k}\rangle$": populations[:, k] for k in [0, 1, 2]},
            figure=fig,)
        for line in fig.axes[0].get_lines():
            line.set_linewidth(1.75)
        ax = fig.axes[0]
        ax.set_xlabel(r"Time ($\mu$s)", fontsize=25)
        ax.set_ylabel("Population", fontsize=25)
        ax.tick_params(axis='both', labelsize=25)
        ax.legend(loc='lower right', bbox_to_anchor=(1, 0.5), fontsize=20)
        plt.tight_layout()
        filename_parts.append(f"{when}.png")

    # Plotting the cost histories
    elif what == "cost":
        fig = plt.figure()
        qv.plot_cost_histories(
        [result["cost_history"]["historical_best"] for result in result.values()],
        labels=[str(learning_rate) for learning_rate in result],
        y_axis_log=True,
        figure=fig,)
        ax = fig.axes[0]
        for line in ax.get_lines():
            line.set_linewidth(1.75)
        ax.set_xlabel(r"Iteration", fontsize=25)
        ax.set_ylabel("Cost", fontsize=25)
        ax.tick_params(axis='both', labelsize=25)
        ncol = min(max_workers, 3)
        bottom_margin = 0.15 + 0.05 * ((max_workers - 1) // ncol)
        ax.legend( loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=ncol, fontsize=11)
        fig.subplots_adjust(bottom=bottom_margin)
        filename_parts.append("cost" + f"_{when}.png")

    # Plotting the pulse shape
    elif what == "controls":
        fig = plt.figure()
        lr = list(result.keys())[0]
        qv.plot_controls({"Pulse 1": result[lr]["output"]["pulse1"], "Pulse 2": result[lr]["output"]["pulse2"],}, 
                         figure=fig)
        filename_parts.append(f"pulses" + f"_{when}.png")
    
    # Saving the figure
    filename = "_".join(filename_parts)
    shape_file_path = os.path.join(shape_path, filename)
    param_file_path = os.path.join(param_path, filename)
    plt.savefig(shape_file_path, dpi=600, bbox_inches='tight')

    # Mr. Clean time
    plt.close(fig)
    plt.clf()
    gc.collect()

### OPTIMIZATION

# Evaluate the efficiency for a quasi-perfect pulse
def evaluate_width(width, pulse_shape, pulse_type):

    # Initialize the graph and calculate the evolution oof the population
    graph = bo.Graph()
    result = calculate_evolution(graph=graph, Delta_signal=const_pwc(graph, Delta),
        delta_signal=const_pwc(graph, delta), Omega1=Omega_1, Omega2=Omega_2,
        pulse_shape=pulse_shape, width=width, sigma_p=0.0, sigma_b=0.0, 
        noise_max=0.0, output_node_names=["states"])
    populations = np.abs(result["output"]["states"]["value"].squeeze()) ** 2

    # Set the tolerance and threshold to the wanted population mismatch
    tolerance = 0.0005
    threshold = 90

    # Calculate the mismatch between populations and the target state
    if pulse_type == "bs":
        end_idx = np.argmax(sample_times >= center_time + 2 * width)
        mismatch = np.abs(populations[end_idx:, 0] - populations[end_idx:, 2])
    elif pulse_type == "m":
        end_idx = np.argmax(sample_times >= center_time + 2 * width)
        mismatch = np.abs(populations[end_idx:, 0])
    
    # If a sufficiently low mismatch is found, return the width and the time
    if np.percentile(mismatch, threshold) < tolerance:
        return width, sample_times[end_idx]
    else:
        return None, None

# Find the correct width for a given pulse shape and type.
def preoptimize_pulse(pulse_shape, pulse_type, initial_width=10e-6):

    # Check if the pulse shape and type are valid
    if pulse_shape not in ["gaus", "box", "sech"]:
        raise ValueError("Implemented shape: gaus, box or sech")
    if pulse_type not in ["bs", "m"]:
        raise ValueError("bs for beam-splitter or m for mirror")
    
    # Initialize the search for an optimal width
    widths = np.arange(initial_width, duration / 2, 0.5 * duration / time_count)
    found_width = None

    # Multiprocessing search to find the optimal width
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_width = {
            executor.submit(evaluate_width, w, pulse_shape, pulse_type): w
            for w in widths}
        for future in as_completed(future_to_width):
            try:
                width, condition_time = future.result()
                if width is not None:
                    # print(f"Preoptimisation completed : {pulse_shape}_{pulse_type} with width: {width*1e6:.4f} µs at time: {condition_time*1e6:.6f} µs \n")
                    for f in future_to_width:
                        if not f.done():
                            f.cancel()
                    found_width = width
                    found_condition_time = condition_time
                    break
            except CancelledError:
                continue
        for future in future_to_width:
            try:
                if not future.done():
                    future.cancel()
                else:
                    future.result() 
            except (CancelledError, Exception):
                pass
    
    # If a valid width is found
    if found_width is not None:
        graph = bo.Graph()
        result = calculate_evolution(graph=graph, Delta_signal=const_pwc(graph, Delta),
            delta_signal=const_pwc(graph, delta), Omega1=Omega_1, Omega2=Omega_2,
            pulse_shape=pulse_shape, width=found_width, sigma_p=0.0, sigma_b=0.0, 
            noise_max=0.0, output_node_names=["states", "unitaries", "pulse1", "pulse2"])
        plotting(result, pulse_shape=pulse_shape, pulse_type=pulse_type, what="states", 
                 when="before")
        plotting(result, pulse_shape=pulse_shape, pulse_type=pulse_type, what="controls",
                 when="before")
        unitaries = result["output"]["unitaries"]["value"]
        target_index = np.searchsorted(sample_times, found_condition_time)
        target_unitary = np.array(unitaries[target_index])

        return found_width, target_unitary, target_index
    
    # If no valid width is found
    print("Condition not reached within max pulse width. Increase Rabi frequencies or duration.")
    return None, None, None

# Single optimization for one learning rate
def single_optimisation(learning_rate, target_unitary, target_index, pulse_shape, pulse_type, sigma_p, sigma_b, noise_max, width, nb_iter):

    # Initialze the graph and the optimization variables
    graph = bo.Graph()
    Omega_1_var = graph.optimizable_scalar(lower_bound=Omega_1 * 0.1, upper_bound=Omega_1 * 10, name="Omega_1",
                                            initial_values=Omega_1)
    Omega_2_var = graph.optimizable_scalar(lower_bound=Omega_2 * 0.1, upper_bound=Omega_2 * 10, name="Omega_2",
                                            initial_values=Omega_2)
    tau = graph.optimizable_scalar(lower_bound=width * 0.1, upper_bound=width * 10, name="tau",
                                    initial_values=width)
    delta_signal = graph.filter_and_resample_pwc(
        pwc=graph.real_optimizable_pwc_signal(segment_count=time_count // 100,
             duration=duration, maximum=delta_max, minimum=0, name="predelta"),
        kernel=graph.sinc_convolution_kernel(1 / width), segment_count=time_count, name="delta")
    
    # Generate the momentum and intensity distributions
    batch_dim = 1
    momentum, batch_dim = momentum_distribution(graph, sigma=sigma_p, batch_dim=batch_dim)
    betas, batch_dim = intensity_distribution(graph, sigma=sigma_b, batch_dim=batch_dim)

    # Generate the phase noise signals
    random_phi_1 = phase_noise(graph, noise_max=noise_max)
    random_phi_2 = phase_noise(graph, noise_max=noise_max)

    # Create the laser pulses
    pulse1 = pulse(graph, pulse_shape, tau, Omega_1_var, name="pulse1")
    pulse2 = pulse(graph, pulse_shape, tau, Omega_2_var, name="pulse2")

    # Set the Hamiltonian and calculate the unitaries
    hamiltonian = set_hamiltonian(graph, const_pwc(graph, Delta), delta_signal,
                                  pulse1, pulse2, momentum, betas, random_phi_1, random_phi_2)
    unitaries = calculate_unitary(graph, hamiltonian)

    # Reduce the dimensionality of the unitaries
    for non_null in [sigma_p != 0.0, sigma_b != 0.0]:
        if non_null:
            unitaries = graph.sum(unitaries, axis=0)
    unitaries = unitaries / ((momentum_batch if sigma_p else 1) *
                                      (intensity_batch if sigma_b else 1))
    
    # Calculate the cost
    if cost_type == "unitaries":
        # target_index = np.min(np.searchsorted(sample_times, center_time + 2 * tau), time_count - 1)
        unitary = calculate_unitary(graph, hamiltonian)[target_index]
        cost = calculate_cost_unitaries(graph, unitary, target_unitary)
    elif cost_type == "states":
        cost = calculate_cost_states(graph, pulse_type, unitaries)

    # Run the optimization using an Adam optimizer 
    result = bo.run_stochastic_optimization(
        graph=graph,
        cost_node_name="cost",
        output_node_names=["Omega_1", "Omega_2", "tau", "delta", "unitaries", "cost", "pulse1", "pulse2"],
        optimizer=bo.stochastic.Adam(learning_rate),
        cost_history_scope="HISTORICAL_BEST",
        iteration_count=nb_iter)
    
    # Mr. Clean time
    del graph, hamiltonian, unitaries, cost
    gc.collect()

    return learning_rate, result

# Main function to optimize the pulse
def optimize_pulse(pulse_shape="gaus", pulse_type="bs", sigma_p=0.0, sigma_b=0.0,
                   noise_max=0.0, width=None, target_unitary=None, target_index=None):
    
    # Find the optimal width and target unitary for the ideal system if not provided
    if None in (width, target_unitary, target_index):
        w, t_u, t_i = preoptimize_pulse(pulse_shape=pulse_shape, pulse_type=pulse_type)
        width = width or w
        target_unitary = target_unitary or t_u
        target_index = target_index or t_i

    # Calculate the evolution of the real system
    if sigma_p > 0.0 or sigma_b > 0.0 or noise_max > 0.0:
        graph = bo.Graph()
        result = calculate_evolution(graph=graph, Delta_signal=const_pwc(graph, Delta),
            delta_signal=const_pwc(graph, delta), Omega1=Omega_1, Omega2=Omega_2,
            pulse_shape=pulse_shape, width=width, sigma_p=sigma_p, sigma_b=sigma_b, 
            noise_max=noise_max, output_node_names=["states", "unitaries"])
        plotting(result, sigma_p=sigma_p, sigma_b=sigma_b, noise_max=noise_max,
                 pulse_shape=pulse_shape, pulse_type=pulse_type, what="states", when="before")
    
    # print("Starting optimization...")
    result = {}

    # Multiprocessing optimization for different learning rates
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_lr = {executor.submit(single_optimisation, lr, target_unitary, target_index, pulse_shape, pulse_type, sigma_p, sigma_b,
                            noise_max, width, nb_iterations): 
                            lr for lr in learning_rates}
        for future in as_completed(future_to_lr):
            try:
                lr, res = future.result()
                result[lr] = res
            except Exception as e:
                print(f"Exception in future for lr={future_to_lr[future]}: {e}")
                traceback.print_exc()
        for future in future_to_lr:
            try:
                if not future.done():
                    future.cancel()
                else:
                    future.result()
            except (CancelledError, Exception):
                pass
    time.sleep(1)

    # Find the best learning rate based on the cost
    best_lr = min(result, key=lambda lr: result[lr]["cost"])
    # print(f"Best learning rate found: {best_lr} with cost = {result[best_lr]['cost']:.4e}")

    # Plot the cost history for all learning rates
    plotting(result, sigma_p=sigma_p, sigma_b=sigma_b, noise_max=noise_max,
             pulse_shape=pulse_shape, pulse_type=pulse_type, what="cost")
    
    # Restart the optimization with the best learning rate
    lr, final_result = single_optimisation(
        best_lr, pulse_shape, pulse_type, sigma_p, sigma_b,
        noise_max, width, 2 * nb_iterations)
    
    # Retrive all the values from the final result
    plotting({best_lr: final_result}, sigma_p=sigma_p, sigma_b=sigma_b, noise_max=noise_max,
             pulse_shape=pulse_shape, pulse_type=pulse_type, what="controls", when="after")
    plotting({best_lr: final_result}, sigma_p=sigma_p, sigma_b=sigma_b, noise_max=noise_max,
             pulse_shape=pulse_shape, pulse_type=pulse_type, what="cost", when="best_lr")
    Omega_1_opt = final_result["output"]["Omega_1"]["value"]
    Omega_2_opt = final_result["output"]["Omega_2"]["value"]
    tau_opt = final_result["output"]["tau"]["value"]
    delta_opt = final_result["output"]["delta"]["values"]

    # Save the optimized parameters to a file
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/params_{pulse_type}_{pulse_shape}_p{sigma_p}_b{sigma_b}_n{noise_max}.pkl"
    with open(filename, "wb") as f:pickle.dump({"Omega_1": Omega_1_opt,"Omega_2": Omega_2_opt,
            "tau": tau_opt,"delta": delta_opt, "learning_rate": best_lr}, f)
    # print(f"Saved optimized parameters to {filename}")

    # Calculate the evolution of the system with the optimized parameters
    graph = bo.Graph()
    result_evol = calculate_evolution(graph=graph, Delta_signal=const_pwc(graph, Delta),
        delta_signal=graph.pwc_signal(values=delta_opt, duration=duration),
        Omega1=Omega_1_opt, Omega2=Omega_2_opt, pulse_shape=pulse_shape, width=tau_opt,
        sigma_p=sigma_p, sigma_b=sigma_b, noise_max=noise_max,
        output_node_names=["states", "unitaries"])
    plotting(result_evol, sigma_p=sigma_p, sigma_b=sigma_b, noise_max=noise_max,
             pulse_shape=pulse_shape, pulse_type=pulse_type, what="states",
             when=f"after_lr_{best_lr}")
    
    # Mr. Clean time
    del result_evol, graph, final_result, result
    gc.collect()
    plt.close('all')
    time.sleep(1)

### MAIN EXECUTION

# Estimate the number of workers based on available resources (NOT IMPLEMENTED)
def estimate_number_workers(time_count, nb_iterations):
    available_memory = psutil.virtual_memory().available
    max_workers = cpu_count() // 2
    print("The logic for it has not been implemented yet and is using default value.") 
    return min(cpu_count-2, max_workers)

# Initialize the optimization parameters
def initialize_optimization(lr_log_min, lr_log_max):
    global max_workers, learning_rates
    max_workers = estimate_number_workers(time_count, nb_iterations)
    learning_rates = np.logspace(lr_log_min, lr_log_max, num=max_workers, base=10.0) 

# Force cleanup of resources to avoid memory leaks
def force_cleanup():
    plt.close('all')
    for _ in range(3):
        gc.collect()
    try:
        for child in mp.active_children():
            try:
                child.terminate()
                child.join(timeout=1)
                if child.is_alive():
                    child.kill()
            except:
                pass
    except:
        pass
    time.sleep(1)

# Simplify the execution for all pulses
def run_all_pulses(sigma_p, sigma_b, noise_max):
    start = time.time()
    for shape in ["gaus", "box", "sech"]:
        for p_type in ["bs", "m"]:
            optimize_pulse(pulse_shape=shape,pulse_type=p_type,
                sigma_p=sigma_p,sigma_b=sigma_b,
                noise_max=noise_max)
            gc.collect()
    print(f"Execution completed in {time.time() - start:.2f} seconds.")
            
# Main part of the script           
if __name__ == "__main__":

    # Set up signal handlers for cleanup on termination
    def signal_handler(signum, frame):
        force_cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start of the calculations
    total_start = time.time()
    initialize_optimization(lr_log_min=-3, lr_log_max=0)


    # Momentum optimization
    run_all_pulses(sigma_p=0.01, sigma_b=0.0, noise_max=0.0)
    run_all_pulses(sigma_p=0.1, sigma_b=0.0, noise_max=0.0)
    run_all_pulses(sigma_p=0.3, sigma_b=0.0, noise_max=0.0)

    # Intensity optimization
    run_all_pulses(sigma_p=0.0, sigma_b=0.3, noise_max=0.0)
    run_all_pulses(sigma_p=0.0, sigma_b=1, noise_max=0.0)
    run_all_pulses(sigma_p=0.0, sigma_b=3, noise_max=0.0)

    # Phase noise optimization
    run_all_pulses(sigma_p=0.0, sigma_b=0.0, noise_max=1e-6)
    run_all_pulses(sigma_p=0.0, sigma_b=0.0, noise_max=1e-4)
    run_all_pulses(sigma_p=0.0, sigma_b=0.0, noise_max=1e-2)

    # Both momentum and phase noise optimization
    run_all_pulses(sigma_p=0.3, sigma_b=0.0, noise_max=1e-4)

    # Both intensity and phase noise optimization
    run_all_pulses(sigma_p=0.0, sigma_b=0.3, noise_max=1e-4)


    # End of the calculations
    force_cleanup()
    print("Total execution time for all pulses: {:.2f} seconds.".format(time.time() - total_start))
