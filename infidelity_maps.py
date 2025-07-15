import gc
import sys
import time
import signal
import matplotlib
import numpy as np

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
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tqdm import tqdm
from itertools import product
from scipy.ndimage import gaussian_filter
from multiprocessing import cpu_count, set_start_method
from tqdm.contrib.concurrent import process_map

plt.style.use(qv.get_qctrl_style())
matplotlib.use('Agg')

# Authenticate (cloud)
# bo.cloud.set_verbosity("QUIET")
# add you key here

# Authenticate (local)
bo.set_local_mode()
unmute_output()

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass 

### CONSTANTS 
hbar = 1.054571817e-34

### ATOM PARAMETERS
m = 86.9092 * 1.6605e-27
k_eff = 2 * np.pi / 780e-9
p_0 = 0.0

### LASER PARAMETERS
Omega_1 = 2 * np.pi * 2.0e6
Omega_2 = 2 * np.pi * 2.0e6
Delta = 2 * np.pi * 1.0e8
delta =  2 * np.pi * 0.0
phi_1 = 0.0
phi_2 = 0.0

### TIME PARAMETERS 
time_count = 5000
duration = 500e-6
center_time = duration / 2
sample_times = np.linspace(0, duration, time_count)
bs_duration = 14.1e-6
m_duration = 27.6e-6

### MATRICES INITIALIZATION
identity1 = np.diag([1, 0, 0])
identity2 = np.diag([0, 1, 0])
identity3 = np.diag([0, 0, 1])
sigma_m12 = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
sigma_m23 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])

### PULSES
def pulse(graph, width, amplitude):
    return graph.signals.gaussian_pulse_pwc(
        duration=duration, segment_count=time_count,
        amplitude=amplitude, width=width, center_time=center_time)

### PHASE NOISE
def phase_noise(graph, noise_max, alpha=1, cutoff=5e4):
    if noise_max <= 0.0:
        return graph.constant_pwc(constant=0.0, duration=duration)
    dt = duration / time_count
    fs = 1 / dt
    pink_noise = cn.powerlaw_psd_gaussian(exponent=alpha, size=time_count)
    freqs = np.fft.rfftfreq(time_count, d=dt)
    spectrum = np.fft.rfft(pink_noise)
    scaling = np.ones_like(freqs)
    nonzero = freqs > 0
    scaling[nonzero] = np.where(
        freqs[nonzero] < cutoff,
        1.0,
        (cutoff / freqs[nonzero]) ** alpha)
    scaling[0] = 0.0
    spectrum *= scaling
    psd = (np.abs(spectrum)**2) / (fs * time_count)
    current_max_psd = np.max(psd)
    if current_max_psd == 0:
        scale_factor = 0.0
    else:
        scale_factor = np.sqrt(noise_max / current_max_psd)
    spectrum *= scale_factor
    hybrid_noise = np.fft.irfft(spectrum, n=time_count)
    return graph.pwc_signal(duration=duration, values=hybrid_noise)

### SIGNALS
def const_pwc(graph, value):
    return graph.constant_pwc(constant=value, duration=duration)

### HAMILTONIAN
def set_hamiltonian(graph, Delta, delta, pulse1, pulse2, 
                    p, betas, random_phi_1, random_phi_2):
    momentum = p  * hbar * k_eff**2 / m
    return (
        graph.hermitian_part(delta * identity1)
        + graph.hermitian_part(-momentum * identity1)
        + graph.hermitian_part((delta + 2 * Delta) * identity2)
        + graph.hermitian_part(-delta * identity3)
        + graph.hermitian_part(momentum * identity3)
        + graph.hermitian_part(betas * pulse1 * sigma_m12 * 
                               graph.exp(phi_1 + random_phi_1))
        + graph.hermitian_part(betas * pulse2 * sigma_m23 * 
                               graph.exp(phi_2 + random_phi_2)))

# UNITARY
def calculate_unitary(hamiltonian, graph):
    return graph.time_evolution_operators_pwc(hamiltonian=hamiltonian, sample_times=sample_times)

# INFIDELITY

# Compute the infidelity between the target and real unitary
def gate_infidelity(target, real, graph):
    return 1 - graph.abs(graph.trace(graph.adjoint(target) @ real) /
                         graph.trace(graph.adjoint(target) @ target)) ** 2

# Compute the infidelity for a given pulse type and parameters
def compute_infidelity(pulse_type, p=0.0, beta=1.0, noise=0.0):
    graph = bo.Graph()
    width = bs_duration if pulse_type == "bs" else m_duration
    pulse1 = pulse(graph, width, Omega_1)
    pulse2 = pulse(graph, width, Omega_2)
    random_phi_1 = phase_noise(graph, noise)
    random_phi_2 = phase_noise(graph, noise)
    hamiltonian_target = set_hamiltonian(graph, const_pwc(graph, Delta), const_pwc(graph, delta), 
                                         pulse1, pulse2, const_pwc(graph, 0.0),
                                         const_pwc(graph, 1.0), 
                                         const_pwc(graph,0.0), const_pwc(graph, 0.0))
    hamiltonian_real   = set_hamiltonian(graph, const_pwc(graph, Delta), const_pwc(graph, delta),  
                                         pulse1, pulse2, const_pwc(graph, p),
                                         const_pwc(graph, beta), random_phi_1, random_phi_2)
    unitary_target = calculate_unitary(hamiltonian_target, graph)
    unitary_real = calculate_unitary(hamiltonian_real, graph)
    infid = gate_infidelity(unitary_target, unitary_real, graph)
    infid.name = "infidelity"
    result = bo.execute_graph(graph, output_node_names=["infidelity"])
    infidelity = np.maximum(0, result["output"]["infidelity"]["value"])
    min_infidelity = np.min(infidelity[time_count//2:])

    # Mr. Clean time
    del graph, hamiltonian_target, hamiltonian_real, unitary_target, unitary_real, infid, result, infidelity
    gc.collect()

    return min_infidelity

# Compute infidelity maps for momentum and phase noise (mn)
def compute_single_infidelity_mn(args):
    i, j, pulse_type, p_values, noise_values = args
    p = p_values[i]
    noise = noise_values[j]
    try:
        val = compute_infidelity(pulse_type, p=p, noise=noise)
    except Exception as e:
        val = np.nan
    return i, j, val

# Compute infidelity maps for momentum and beta (mb)
def compute_single_infidelity_mb(args):
    i, j, pulse_type, p_values, b_values = args
    p = p_values[i]
    beta = b_values[j]
    try:
        val = compute_infidelity(pulse_type, p=p, beta=beta)
    except Exception as e:
        val = np.nan
    return i, j, val

# Compute infidelity maps for beta and phase noise (bn)
def compute_single_infidelity_bn(args):
    i, j, pulse_type, b_values, noise_values = args
    beta = b_values[i]
    noise = noise_values[j]
    try:
        val = compute_infidelity(pulse_type, beta=beta, noise=noise)
    except Exception as e:
        val = np.nan
    return i, j, val

# Compute infidelity maps for momentum and noise (mn)
def mn_map(pulse_type, n):

    # Choose a range of values for momentum and noise
    p_values = np.logspace(-6, 0, n)
    noise_values = np.logspace(-12, -3, n)

    # Create a grid of indices for the parameter combinations
    all_indices = list(product(range(len(p_values)), range(len(noise_values))))
    args_list = [(i, j, pulse_type, p_values, noise_values) for i, j in all_indices]
    infidelity_map = np.zeros((len(p_values), len(noise_values)))

    # Compute the infidelities in parallel
    results = process_map(
        compute_single_infidelity_mn,
        args_list,
        max_workers=cpu_count()-1,
        chunksize=1,
        desc="Computing Infidelities")
    for i, j, val in results:
        infidelity_map[i, j] = val

    # Mr. Clean time
    del results, args_list, all_indices
    gc.collect()

    # Save the infidelity map
    infidelity_map = gaussian_filter(infidelity_map, sigma=2, mode='nearest')
    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(noise_values, p_values)
    c = ax.pcolormesh(
        X, Y, infidelity_map, shading='gouraud', cmap='viridis',
        norm=mcolors.LogNorm(vmin=np.maximum(1e-9, np.min(infidelity_map[infidelity_map > 0])), vmax=np.max(infidelity_map)))
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Infidelity", fontsize=20)
    ax.set_xlabel(r"Phase noise spectral density maximum ($\mathrm{rad}^2/\mathrm{Hz}$)", fontsize=20)
    ax.set_ylabel(r"Momentum ($\hbar k_{\mathrm{eff}}$)", fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"plots/infidelity_map_mn_{pulse_type}.png", dpi=500, bbox_inches='tight')

    # Mr. Clean time
    plt.close(fig)
    plt.clf()
    del infidelity_map, X, Y, p_values, noise_values
    gc.collect()

# Compute infidelity maps for momentum and beta (mb)
def mb_map(pulse_type, n):

    # Choose a range of values for momentum and beta
    p_values = np.logspace(-6, 0, n)
    b_values = np.linspace(0.01, 1, n)

    # Create a grid of indices for the parameter combinations
    all_indices = list(product(range(len(p_values)), range(len(b_values))))
    args_list = [(i, j, pulse_type, p_values, b_values) for i, j in all_indices]
    infidelity_map = np.zeros((len(p_values), len(b_values)))

    # Compute the infidelities in parallel
    results = process_map(
        compute_single_infidelity_mb,
        args_list,
        max_workers=cpu_count()-1,
        chunksize=1,
        desc="Computing Infidelities")
    for i, j, val in results:
        infidelity_map[i, j] = val
    
    # Mr. Clean time
    del results, args_list, all_indices
    gc.collect()

    # Save the infidelity map
    infidelity_map = gaussian_filter(infidelity_map, sigma=2, mode='nearest')
    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(b_values, p_values)
    c = ax.pcolormesh(
        X, Y, infidelity_map, shading='gouraud', cmap='viridis',
        norm=mcolors.LogNorm(vmin=np.maximum(1e-9, np.min(infidelity_map[infidelity_map > 0])), vmax=np.max(infidelity_map)))
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Infidelity", fontsize=20)
    ax.set_xlabel(r"Beta", fontsize=20)
    ax.set_ylabel(r"Momentum ($\hbar k_{\mathrm{eff}}$)", fontsize=20)
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"plots/infidelity_map_mb_{pulse_type}.png", dpi=500, bbox_inches='tight')

    # Mr. Clean time
    plt.close(fig)
    plt.clf()
    del infidelity_map, X, Y, p_values, b_values
    gc.collect()

# Compute infidelity maps for beta and phase noise (bn)
def bn_map(pulse_type, n): 

    # Choose a range of values for beta and noise
    b_values = np.linspace(0.01, 1, n)
    noise_values = np.logspace(-12, -3, n)

    # Create a grid of indices for the parameter combinations
    all_indices = list(product(range(len(b_values)), range(len(noise_values))))
    args_list = [(i, j, pulse_type, b_values, noise_values) for i, j in all_indices]
    infidelity_map = np.zeros((len(b_values), len(noise_values)))

    # Compute the infidelities in parallel
    results = process_map(
        compute_single_infidelity_bn,
        args_list,
        max_workers=cpu_count()-1,
        chunksize=1,
        desc="Computing Infidelities")
    for i, j, val in results:
        infidelity_map[i, j] = val

    # Mr. Clean time
    del results, args_list, all_indices
    gc.collect()

    # Save the infidelity map
    infidelity_map = gaussian_filter(infidelity_map, sigma=2, mode='nearest')
    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(noise_values, b_values)
    c = ax.pcolormesh(
        X, Y, infidelity_map, shading='gouraud', cmap='viridis',
        norm=mcolors.LogNorm(vmin=np.maximum(1e-9, np.min(infidelity_map[infidelity_map > 0])), vmax=np.max(infidelity_map)))
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Infidelity", fontsize=20)
    ax.set_xlabel(r"Phase noise spectral density maximum ($\mathrm{rad}^2/\mathrm{Hz}$)", fontsize=20)
    ax.set_ylabel(r"Beta", fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"plots/infidelity_map_bn_{pulse_type}.png", dpi=500, bbox_inches='tight')

    # Mr. Clean time
    plt.close(fig)
    plt.clf()
    del infidelity_map, X, Y, b_values, noise_values
    gc.collect()

### MAIN EXECUTION

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

# Main part of the script
if __name__ == "__main__":

    # Set up signal handlers for cleanup on termination
    def signal_handler(signum, frame):
        force_cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start of execution
    start_time = time.time()
    
    n = 50 # Number of points in the grid
    pulse_type = "bs"
    mn_map(pulse_type, n)
    gc.collect()
    mb_map(pulse_type, n)
    gc.collect()
    bn_map(pulse_type, n)
    gc.collect()
    
    pulse_type = "m"
    mn_map(pulse_type, n)
    gc.collect()
    mb_map(pulse_type, n)
    gc.collect()
    bn_map(pulse_type, n)
    gc.collect()

    # End of execution
    force_cleanup()
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
