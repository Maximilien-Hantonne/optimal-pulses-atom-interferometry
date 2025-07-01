# Quantum Pulse Optimization for Cold Atom Interferometry

This repository contains two main Python scripts for quantum pulse optimization and analysis in cold atom interferometry systems, specifically designed for Rubidium-87 atomic interferometry using three-level system.

## 📁 Files Overview

- **`pulse_opt.py`**: Main optimization script for quantum pulse parameters
- **`infidelity_map.py`**: Analysis script for generating infidelity maps across parameter spaces
- **`README.md`**: This documentation file

## 🔧 Requirements

### Python Dependencies

Install the required packages using pip:

```bash
pip install numpy scipy matplotlib tqdm colorednoise
```

### External Libraries

#### Boulder Opal (Proprietary)
The primary quantum control library for pulse optimization:

**⚠️ Proprietary Software - License Required**

```bash
# For cloud version (requires API key)
pip install boulderopal

# For local version
pip install boulderopal[local]
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
- **CPU**: Multi-core processor (the scripts utilize parallel processing)
- **Disk Space**: Several GB for output plots and data files

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
Optimizes quantum pulse parameters to achieve high-fidelity beam-splitter and mirror operations for atomic interferometry.

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
- **Intensity fluctuations**: `sigma_b` (laser intensity variations)
- **Phase noise**: `noise_max` (laser phase instability, max PSD in rad²/Hz)

### Optimization Process

1. **Pre-optimization**: Finds initial pulse width for target gate operation
2. **Parallel optimization**: Tests multiple learning rates simultaneously using `ProcessPoolExecutor`
3. **Parameter optimization**: 
   - Rabi frequencies (Ω₁, Ω₂)
   - Pulse duration (τ)
   - Two-photon detuning profile (δ(t))
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

### Usage Example

```python
# Run all pulse shapes and types with momentum spread
run_all_pulses(sigma_p=0.1, sigma_b=0.0, noise_max=0.0)

# Single optimization
optimize_pulse(pulse_shape="gaus", pulse_type="bs", 
               sigma_p=0.01, sigma_b=0.0, noise_max=0.0)
```

## 📈 infidelity_map.py - Infidelity Analysis

### Purpose
Generates 2D parameter maps showing gate infidelity across different noise conditions to understand robustness and identify optimal operating regions.

### Map Types

#### 1. Momentum-Noise Maps (`mn_map`)
- **X-axis**: Phase noise spectral density maximum (rad²/Hz)
- **Y-axis**: Momentum spread (ℏk_eff units)
- **Shows**: How phase noise and atomic motion affect gate fidelity

#### 2. Momentum-Beta Maps (`mb_map`)
- **X-axis**: Beta (intensity factor, 0-1)
- **Y-axis**: Momentum spread (ℏk_eff units)  
- **Shows**: Impact of laser intensity variations and atomic motion

#### 3. Beta-Noise Maps (`bn_map`)
- **X-axis**: Phase noise spectral density maximum (rad²/Hz)
- **Y-axis**: Beta (intensity factor, 0-1)
- **Shows**: Combined effect of intensity and phase fluctuations

### Key Features

- **Parallel computation**: Uses `process_map` for efficient parameter space exploration
- **Gaussian filtering**: Smooths infidelity maps (σ=2)
- **Logarithmic visualization**: Uses `LogNorm` for better dynamic range
- **High resolution**: 50×50 parameter grids by default

### Output
- **Location**: `plots/` directory
- **Files**: `infidelity_map_{type}_{pulse_type}.png`
- **Format**: High-resolution PNG (500 DPI)

### Usage Example

```python
# Generate all maps for beam-splitter with 50×50 resolution
n = 50
mn_map("bs", n)  # Momentum vs Noise
mb_map("bs", n)  # Momentum vs Beta  
bn_map("bs", n)  # Beta vs Noise
```

## ⚙️ Configuration Parameters

### Physical Constants
```python
# Rubidium-87 parameters
m_Rb = 86.9092 * 1.6605e-27      # Atomic mass
omega_0 = 2 * np.pi * c / 780.241209e-9    # Transition frequency
k_eff = 2 * np.pi / 780e-9                 # Effective wave vector

# Laser parameters  
Omega_1 = Omega_2 = 2 * np.pi * 2.0e6   # Rabi frequencies (rad/s)
Delta = 2 * np.pi * 1.0e8               # Single-photon detuning (rad/s)
```

### Computational Parameters
```python
time_count = 2500                 # Time discretization points
duration = 500e-6                 # Total pulse duration (s)
max_workers = cpu_count() // 2    # Parallel workers
nb_iterations = 20                # Optimization iterations
```

## 🔄 Process Management

Both scripts include comprehensive CPU process cleanup to prevent lingering processes:

### Features
- **Signal handling**: Graceful shutdown on Ctrl+C
- **Process termination**: Automatic cleanup of worker processes
- **Memory management**: Explicit garbage collection
- **Resource cleanup**: Matplotlib figure and Boulder Opal graph cleanup

### Force Cleanup
If processes persist after execution, the scripts include `force_cleanup()` functions that:
- Terminate all child processes
- Clear matplotlib figures
- Force garbage collection
- Kill unresponsive processes

## 📊 Typical Workflow

1. **Initial optimization**:
   ```python
   python pulse_opt.py
   ```

2. **Analyze robustness**:
   ```python
   python infidelity_map.py
   ```

3. **Review outputs**:
   - Check `plots/` for visualizations
   - Load optimized parameters from `data/` directory
   - Compare different pulse shapes and noise conditions

## 🎯 Performance Tips

- **Memory**: Use smaller parameter grids (`n < 50`) for initial testing
- **CPU usage**: Adjust `max_workers` based on available cores
- **Disk space**: Large parameter sweeps can generate many plot files
- **Optimization time**: Typical runs take 30 minutes to several hours

## 📝 Example Parameter Sets

### Conservative (fast testing)
```python
sigma_p = 0.01      # Small momentum spread
sigma_b = 0.0       # No intensity noise  
noise_max = 0.0     # No phase noise
n = 25              # Coarse parameter grid
```

### Realistic experimental conditions
```python
sigma_p = 0.1       # Moderate momentum spread
sigma_b = 0.3       # 30% intensity variations
noise_max = 1e-4    # Typical phase noise level
n = 50              # High-resolution maps
```

### Challenging conditions
```python
sigma_p = 0.3       # Large momentum spread
sigma_b = 1.0       # Strong intensity variations  
noise_max = 1e-2    # High phase noise
n = 75              # Very high resolution
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `time_count` or parameter grid size `n`
2. **Slow optimization**: Decrease `nb_iterations` or use fewer workers
3. **Lingering processes**: Scripts include automatic cleanup, but you can manually run `force_cleanup()`
4. **Import errors**: Ensure all dependencies are installed

### Performance Optimization

- Use local Boulder Opal mode for faster execution
- Adjust `chunksize` in `process_map` for better load balancing
- Monitor CPU and memory usage during large parameter sweeps

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

This MIT license applies **ONLY** to the code in this repository (`pulse_opt.py`, `infidelity_map.py`, and associated scripts). It does **NOT** grant any rights to the following proprietary software components:

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

## �📚 References

This code implements quantum optimal control techniques for atomic interferometry as described in quantum control and atomic physics literature. The optimization uses gradient-based methods through the Boulder Opal framework for efficient quantum control pulse design.
