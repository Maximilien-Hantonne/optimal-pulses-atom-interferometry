# Quantum Pulse Optimization for Atom Interferometry

This repository contains two main Python scripts for quantum pulse optimization and analysis in cold atom interferometry systems, specifically designed for Rubidium-87 atomic interferometry using three-level system.

## 📁 Files Overview

- **`pulse_opt.py`**: Main optimization script for quantum pulse parameters
- **`pulse_opt_doc.py`**: Comprehensive documentation and helper functions for pulse optimization
- **`infidelity_map.py`**: Analysis script for generating infidelity maps across parameter spaces
- **`README.md`**: This documentation file

## 🔧 Requirements

#### Boulder Opal (Proprietary)
The primary quantum control library for pulse optimization:

**⚠️ Proprietary Software - License Required**

```bash
pip install boulderopal
```

**Note**: Boulder Opal is proprietary software owned by QCtrl. You must obtain a valid license from QCtrl to use this software.

#### QCtrl Visualizer (Proprietary)
For plotting and visualization:

**⚠️ Proprietary Software - License Required**

```bash
pip install qctrlvisualizer
```

**Note**: QCtrl Visualizer is proprietary software owned by QCtrl. You must obtain a valid license from QCtrl to use this software.

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large parameter sweeps)
- **CPU**: Multi-core processor (both scripts utilize parallel processing)
- **Disk Space**: Several GB for output plots and data files

### Python Dependencies

Install the required packages using pip:

```bash
pip install numpy scipy matplotlib tqdm colorednoise
```

## 🧮 Physics Background

### Three-Level Lambda System

Both scripts simulate a three-level lambda system in Rubidium-87:
- **State |1⟩**: Ground state F=1
- **State |2⟩**: Excited state 
- **State |3⟩**: Ground state F=2

### Hamiltonian

The system Hamiltonian includes:
- **Momentum dependence**: Accounts for atomic motion and recoil
- **Laser interactions**: Two-photon Raman transitions
- **Detunings**: Single-photon (Δ) and two-photon (δ) detunings
- **Noise sources**: Phase noise, intensity fluctuations, momentum spread

## 📊 pulse_opt.py - Pulse Optimization

### Purpose
Optimizes quantum pulse parameters to achieve high-fidelity beam-splitter and mirror operations for atomic interferometry using gradient-based optimization through Boulder Opal.

### Key Parameters

#### Physical Constants
```python
# Rubidium-87 parameters
m_Rb = 86.9092 * 1.6605e-27      # Atomic mass (kg)
omega_0 = 2 * np.pi * c / 780.241209e-9    # Transition frequency (rad/s)
k_eff = 2 * np.pi / 780e-9                 # Effective wave vector (1/m)

# Laser parameters  
Omega_1 = Omega_2 = 2 * np.pi * 2.0e6   # Rabi frequencies (rad/s)
Delta = 2 * np.pi * 1.0e8               # Single-photon detuning (rad/s)
```

### Documentation and Helper Functions
The main script is supported by comprehensive documentation in `pulse_opt_doc.py` which provides:

- **Function Documentation**: Documentation for all functions
- **Helper Functions**: Utilities for parameter sweeps, time estimation, and experiment planning
- **System Information**: Physical system overview and troubleshooting guides

#### Using the Documentation Functions

```python
from pulse_opt_doc import get_function_help, print_system_overview, print_troubleshooting

# Get function documentation  
help_text = get_function_help("calculate_evolution")
print(help_text)

# Get system information
print_system_overview()      # Physical system overview
print_troubleshooting()      # Problem-solving guide

# Get help for specific functions
print(get_function_help("optimize_pulse"))
print(get_function_help("preoptimize_pulse"))
print(get_function_help("momentum_distribution"))
```

### Key Features

#### Pulse Shapes
- **Gaussian**: `pulse_shape="gaus"`
- **Box**: `pulse_shape="box"` 
- **Hyperbolic Secant**: `pulse_shape="sech"`

#### Gate Types
- **Beam-splitter**: `pulse_type="bs"` (creates superposition states)
- **Mirror**: `pulse_type="m"` (reflects atomic wavefunction)

#### Noise Sources
- **Momentum spread**: `sigma_p` (thermal motion, σ in units of ℏk_eff)
- **Intensity fluctuations**: `sigma_b` (laser intensity variations, dimensionless)
- **Phase noise**: `noise_max` (laser phase instability, max PSD in rad²/Hz)

### Optimization Process

1. **Pre-optimization**: Finds initial pulse width for target gate operation
2. **Parallel optimization**: Tests multiple learning rates simultaneously using `ProcessPoolExecutor`
3. **Parameter optimization**: Rabi frequencies (Ω₁, Ω₂), pulse duration (τ), two-photon detuning profile (δ(t))
4. **Final optimization**: Extended optimization with best learning rate

### Output Files

#### Plots Directory Structure
```
plots/
├── sorted_plots/
│   ├── bs/gaus/perfect/           # Beam-splitter, Gaussian, no noise
│   ├── bs/gaus/p_0.01/           # Beam-splitter, Gaussian, momentum spread
│   └── ...
└── parameters/
    ├── perfect/                   # Cross-comparison by parameter set
    ├── p_0.01/
    └── ...
```

#### Data Files
- **Location**: `data/` directory
- **Format**: Pickle files (`.pkl`)
- **Content**: Optimized parameters (Ω₁, Ω₂, τ, δ(t), learning_rate)
- **Naming**: `params_{pulse_type}_{pulse_shape}_p{sigma_p}_b{sigma_b}_n{noise_max}.pkl`


## 📈 infidelity_map.py - Infidelity Analysis

### Purpose
Generates 2D infidelity heatmaps across different noise parameter spaces to analyze gate robustness and identify optimal operating regions. Uses pre-computed optimal pulse durations for beam-splitter and mirror operations.

### Key Parameters 

#### Physical Constants
```python
# Rubidium-87 parameters
m = 86.9092 * 1.6605e-27        # Atomic mass (kg)
k_eff = 2 * np.pi / 780e-9      # Effective wave vector (1/m)

# Laser parameters
Omega_1 = Omega_2 = 2 * np.pi * 2.0e6   # Rabi frequencies (rad/s)
Delta = 2 * np.pi * 1.0e8               # Single-photon detuning (rad/s)
delta = 2 * np.pi * 0.0                 # Two-photon detuning (rad/s)
```

#### Parameter Ranges
```python
# Momentum spread values (logarithmic scale)
p_values = np.logspace(-6, 0, n)         # 10^-6 to 1 ℏk_eff units

# Intensity variation values (linear scale)  
b_values = np.linspace(0.01, 1, n)       # 0.01 to 1.0 (beta factor)

# Phase noise values (logarithmic scale)
noise_values = np.logspace(-12, -3, n)   # 10^-12 to 10^-3 rad²/Hz
```

### Map Types 

#### 1. Momentum-Noise Maps (`mn_map`)
- **X-axis**: Phase noise spectral density maximum (10^-12 to 10^-3 rad²/Hz)
- **Y-axis**: Momentum spread (10^-6 to 1 ℏk_eff units)
- **Function**: How phase noise and atomic thermal motion affect gate fidelity
- **Use case**: Optimizing laser coherence vs atomic cooling requirements

#### 2. Momentum-Beta Maps (`mb_map`)
- **X-axis**: Beta - intensity factor (0.01 to 1.0, linear scale)
- **Y-axis**: Momentum spread (10^-6 to 1 ℏk_eff units, log scale)
- **Function**: Impact of laser intensity variations and atomic motion
- **Use case**: Beam profile optimization and atomic cloud size effects

#### 3. Beta-Noise Maps (`bn_map`)
- **X-axis**: Phase noise spectral density maximum (10^-12 to 10^-3 rad²/Hz)
- **Y-axis**: Beta - intensity factor (0.01 to 1.0, linear scale)
- **Function**: Combined effect of intensity and phase fluctuations
- **Use case**: Laser system stability requirements

### Key Features

- **Pre-computed pulse durations**: Uses optimized pulse widths
- **Fixed pulse shape**: Gaussian pulses only (optimized for this application)
- **Parallel computation**: Uses `process_map` with `cpu_count()-1` workers
- **Gaussian filtering**: Smooths infidelity maps (σ=2) for visual clarity
- **Logarithmic visualization**: Uses `LogNorm` for better dynamic range display
- **High resolution**: Default 50×50 parameter grids (2500 calculations per map)
- **Automatic cleanup**: Comprehensive memory management and process cleanup

### Analysis Process

1. **Target gate computation**: Calculate ideal unitary evolution (no noise)
2. **Real gate computation**: Calculate actual evolution with specified noise parameters
3. **Gate infidelity calculation**: Compare target vs real using trace fidelity metric
4. **Minimum selection**: Extract minimum infidelity from second half of pulse
5. **Map generation**: Collect results across parameter grid and apply Gaussian smoothing

### Output Files
- **Location**: `plots/` directory
- **Files**: 
  - `infidelity_map_mn_bs.png` (momentum-noise, beam-splitter)
  - `infidelity_map_mb_bs.png` (momentum-beta, beam-splitter)
  - `infidelity_map_bn_bs.png` (beta-noise, beam-splitter)
  - `infidelity_map_mn_m.png` (momentum-noise, mirror)
  - `infidelity_map_mb_m.png` (momentum-beta, mirror)
  - `infidelity_map_bn_m.png` (beta-noise, mirror)
- **Format**: High-resolution PNG (500 DPI)
- **Colormap**: Viridis with logarithmic normalization

## 🎯 Performance Tips

### For pulse_opt.py (Optimization)
- **Memory**: Use smaller `time_count` (try 2000-3000) for initial testing
- **CPU usage**: Adjust `max_workers` based on available cores (default: cpu_count()//2)
- **Optimization time**: Typical runs take 30 min to several hours depending on parameters

### For infidelity_map.py
- **Memory**: Use smaller grid resolution (`n < 50`) for initial testing
- **CPU usage**: Uses `cpu_count()-1` workers automatically
- **Analysis time**: Typical runs take 30-60 minutes for all maps
- **Grid resolution**: 50×50 gives good detail, 75×75 for publication quality

## 📝 Parameter Guidelines

### pulse_opt.py Parameter Ranges

```python
# Momentum spread (thermal motion effects)
sigma_p = 0.0    # Perfect cooling
sigma_p = 0.01   # Good cooling (typical)
sigma_p = 0.1    # Moderate cooling
sigma_p = 0.3    # Poor cooling / hot atoms

# Intensity variations (beam profile effects)  
sigma_b = 0.0    # Perfect uniform beam
sigma_b = 0.3    # 30% intensity variations (typical)
sigma_b = 1.0    # Strong spatial variations

# Phase noise (laser stability)
noise_max = 0.0    # Perfect coherence
noise_max = 1e-6   # Excellent laser (typical)
noise_max = 1e-4   # Good laser stability
noise_max = 1e-2   # Poor laser stability
```

### infidelity_map.py Parameter Ranges

The analysis uses logarithmic and linear parameter sweeps:

```python
# Pre-defined parameter ranges (cannot be changed without editing script)
momentum_range = [1e-6, 1]        # ℏk_eff units (log scale)
intensity_range = [0.01, 1.0]     # Beta factor (linear scale)  
phase_noise_range = [1e-12, 1e-3] # rad²/Hz (log scale)
grid_resolution = 50               # 50×50 = 2500 calculations per map
```


## � License

### MIT License

Copyright (c) 2025 [Maximilien HANTONNE]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Third-Party Dependencies

**⚠️ Important License Notice:**

This MIT license applies **ONLY** to the code in this repository (`pulse_opt.py`, `pulse_opt_doc.py`, `infidelity_map.py`, and associated scripts). It does **NOT** grant any rights to the following proprietary software components:

#### Boulder Opal (QCtrl)
- **Boulder Opal** is proprietary software owned by QCtrl
- **Separate license required** from QCtrl to use Boulder Opal
- Available in both cloud and local versions (each requiring appropriate licensing)
- This repository's MIT license grants **NO RIGHTS** to Boulder Opal software
- Users must obtain their own Boulder Opal license from QCtrl

#### QCtrl Visualizer
- **QCtrl Visualizer** is proprietary software owned by QCtrl  
- **Separate license required** from QCtrl to use QCtrl Visualizer
- This repository's MIT license grants **NO RIGHTS** to QCtrl Visualizer


### Usage Disclaimer

To run the code in this repository, you must:

1. ✅ **Use the MIT-licensed code** in this repository freely
2. 🔐 **Obtain valid licenses** for Boulder Opal and QCtrl Visualizer from QCtrl
3. 📋 **Comply with all license terms** of the proprietary QCtrl software

**The authors of this repository have no affiliation with QCtrl and cannot provide licenses for QCtrl's proprietary software.**
