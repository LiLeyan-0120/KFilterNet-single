# KFilterNet: High Adaptability Trajectory Tracking Algorithm Based on Neural Network-Aided Kalman Filtering

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning implementation of adaptive Kalman filter networks for single-target trajectory tracking in complex sensor fusion environments. This project combines classical Kalman filtering theory with modern deep learning techniques, specifically designed for highly dynamic scenarios with complex motion patterns and time-varying noise conditions.

## Abstract

Traditional Kalman Filter (KF) relies on idealized assumptions of "fixed noise statistics + accurate model prior". However, in complex operational scenarios characterized by:
- Composite maneuvering with "instantaneous high-G maneuver + long-term steady cruise"
- Non-stationary noise from sensors
- Asynchronous data and packet loss

These conditions lead to model mismatch and filter divergence. To address these challenges, KFilterNet proposes a highly adaptable architecture integrating deep learning with traditional filtering, achieving dynamic optimization of core filtering parameters and adaptive estimation of time-varying noise covariance.

## Key Innovations

### 1. Modular Decoupling Design
The network decouples the estimation of five core Kalman filter components into independent functional units:
- **State Transition Matrix (Fk)**: Learns dynamic motion patterns
- **Observation Matrix (Hk)**: Adapts to sensor characteristics
- **Kalman Gain (Kk)**: Optimal gain estimation
- **Process Noise Covariance (Qk)**: Time-varying process noise
- **Measurement Noise Covariance (Rk)**: Adaptive measurement noise

### 2. Historical Context Encoding
Utilizes **Gated Recurrent Units (GRUs)** (`HistoryEncoder` in `models/components/kalman_components.py`) to extract features from historical measurement sequences `Z0:k = [z0, z1, ..., zk]`, enabling the model to learn temporal dependencies and adapt to changing conditions. All other learnable components are MLPs.

### 3. Flexible Configuration Modes
Supports four module configuration types for different operational requirements (`ModuleType` in `constants.py`):
- **FIXED**: classical matrix operations
- **SEMI_FIXED**: base matrix + neural residual learning
- **LEARNABLE**: fully neural network-based prediction
- **HYBRID**: network prediction with mathematical constraints

## 🚀 Features

- **Flexible Module Configurations**: Each of the 5 Kalman filter components (Fk, Hk, Kk, Qk, Rk) can be independently configured as learnable / fixed / semi-fixed / hybrid.

- **Comprehensive Motion Models** — 7 motion modes reachable from every entry point, selected via `DataConfig.motion_modes`:

  | Config key | Description |
  |---|---|
  | `constant_velocity` | Linear propagation, acceleration zeroed each step |
  | `constant_acceleration` | Piecewise-constant acceleration |
  | `coordinated_turn` | Heading-rate rotation in the X–Y plane, centripetal acceleration `a = v·ω` recomputed |
  | `weaving` | Sinusoidal turn rate `ω(t) = amp·sin(2πft)`, f ∈ U(0.2, 0.5) Hz |
  | `vertical_maneuver` | Climb U(30,80)° / dive −U(30,60)°, with `g = 9.8` gravity compensation |
  | `loop` | 360° loop with radius U(500,1000) m and energy-conserving speed decay `v² = v₀² − 2gh` |
  | `maneuvering` | Markov-switching mixture across the 6 modes above, minimum dwell 2.0 s |

- **Layered Noise Models** — deliberately separated by role:
  - **Training-time** (`utils/data_generator.py`, 3 types): Gaussian white noise → impulse noise (electronic-warfare interference, p = 0.05, magnitude ×U(5,20)·σ) → first-order Markov correlated noise `n_k = ρ·n_{k−1} + √(1−ρ²)·w_k` with ρ = 0.4. Impulse noise acts on **observations only**, never on the process model.
  - **Unseen-noise generalization tests** (`robustness_study.py`): Gamma, heavy-tailed Laplace, and 70/30 Gaussian mixtures. **These distributions never appear in training data** — they exist solely to measure generalization.
  - **Unseen-motion generalization tests** (`robustness_study.py`): 3D spiral, jerk motion, impulsive turn.
  - **Noise intensity sweeps** (`sensitivity_study.py`): pure-Gaussian isolation with σ_obs ∈ [0.1 … 10] and σ_proc ∈ [0.01 … 2].

- **Three Dataset Presets** (`--dataset_name`, selected in `training/config.py`):
  - `complex_complex_noise` — all 7 motion modes, impulse + correlated noise, σ randomized per sample
  - `complex_fixed_noise` — all 7 motion modes, Gaussian only with fixed σ
  - `simple_ca_complex_noise` — constant acceleration only, complex noise retained

- **Robust Training**:
  - Numerical stability enforcement
  - Early stopping and learning rate scheduling
  - Gradient clipping and checkpoint management (`--resume` supported)

## 📁 Project Structure

```
KFilterNet-single/
├── models/                             # Model implementations
│   ├── KFilterNet_single.py            # Main KFilterNet model (single fused input)
│   ├── KFilterNet_single_stepInput.py  # Step-input variant
│   ├── components/
│   │   └── kalman_components.py        # MLP, NoiseNetwork, StateTransitionNetwork,
│   │                                   # KalmanGainNetwork, ObservationMatrixNetwork,
│   │                                   # HistoryEncoder (GRU), CovarianceNetwork,
│   │                                   # InitialStateComputer
│   └── baseline/                        # Baseline filters
│       ├── baseline_KF.py               # Standard Kalman Filter
│       ├── baseline_AKF.py              # Adaptive Kalman Filter
│       ├── baseline_IMM.py              # Interacting Multiple Model (CV/CA/CT)
│       ├── baseline_UKF.py              # Unscented Kalman Filter
│       └── kalman_net_tsp.py            # KalmanNet baseline
├── models/pretrained/                   # Pretrained weights
│   ├── KFilterNet.pth
│   └── Kalmannet.pth
├── training/                           # Training utilities
│   ├── config.py                       # DataConfig / ModelConfig
│   ├── trainer.py                      # Training loop
│   └── data_loader.py                  # Data loading
├── utils/                              # Utility functions
│   ├── data_generator.py               # Trajectory & noise generation
│   ├── visualization.py                # Visualization tools
│   └── logger.py                       # Logging setup
├── constants.py                        # Centralized constants
├── main.py                             # Single-model training entry point
├── ablation_study.py                   # Module-configuration ablation (12 runs)
├── comparison_study.py                 # Comparison against baselines
├── robustness_study.py                 # Unseen noise / unseen motion generalization
├── sensitivity_study.py                # Sequence length / sample size / noise intensity
├── outputs/                            # Generated results (git-ignored)
│   ├── Ablation/  Comparison/  Robustness/  Sensitivity/
├── requirements.txt
└── LICENSE
```

> Note: `outputs/`, `data/raw/`, and `*.pth` are excluded by `.gitignore`. Pretrained weights are
> committed explicitly so the studies run out of the box.

## 🚀 Quick Start

### Basic Training

```bash
python main.py
```

Defaults: `--model_type KFilterNet`, `--dataset_name complex_complex_noise`, 150 epochs,
5000/1000/2000 train/val/test samples, seed 42.

### Custom Training

`main.py` accepts 16 arguments:

| Argument | Default | Notes |
|---|---|---|
| `--model_type` | `KFilterNet` | One of `KFilterNet`, `KFilterNet_stepInput`, `kalman_net_tsp` |
| `--dataset_name` | `complex_complex_noise` | See dataset presets above |
| `--epochs` | `150` | |
| `--batch_size` | `32` | |
| `--learning_rate` | `1e-3` | |
| `--hidden_dim` | `128` | |
| `--num_layers` | `3` | |
| `--dropout` | `0.1` | |
| `--device` | `cpu` | e.g. `cuda` |
| `--train_samples` / `--val_samples` / `--test_samples` | `5000` / `1000` / `2000` | |
| `--force_regenerate` | `False` | Regenerate the dataset |
| `--output_dir` / `--experiment_name` | `outputs` / auto | |
| `--resume` | `None` | Checkpoint path |

```bash
python main.py \
    --model_type KFilterNet \
    --dataset_name complex_complex_noise \
    --epochs 150 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --hidden_dim 128 \
    --device cpu
```

> **Module types are not CLI flags.** `H_type`, `K_type`, `F_type`, `Q_type`, `R_type`, and `init_type`
> are fields of `ModelConfig` in `training/config.py`. The defaults are the ablation-winning
> `qr_learnable` configuration: `F`/`H`/`K`/`init` = `fixed`, `Q`/`R` = `learnable`. Edit the config to change them.

### Ablation Study

3 datasets × 4 module configurations = 12 training runs. No CLI arguments.

```bash
python ablation_study.py
```

Configurations: `all_learnable`, `fhqr_semi_fixed`, `qr_learnable`, `fh_learnable`.
Output: `outputs/Ablation/` including `dataset_comparison.{csv,json}` and a report.

### Comparison with Baselines

Compares `KFilterNet`, `KF`, `AKF`, `IMM`, `UKF`, `KalmanNetNN` across motion-mode groups and noise levels. No CLI arguments.

```bash
python comparison_study.py
```

Output: `outputs/Comparison/`, including Nature-styled trajectory figures and Illustrator-editable
`.ai` / `.eps` / `.png` sets under `figure/`.

### Robustness Analysis

Evaluates generalization to **unseen** noise (Gamma, Laplace, mixture-Gaussian) and **unseen** motion
(spiral, jerk, impulsive turn), and reports numerical-health metrics
(`divergence_rate`, `nan_per_million`, `settled_time`). No CLI arguments.

```bash
python robustness_study.py
```

Output: `outputs/Robustness/`.

### Sensitivity Analysis

Sweeps sequence length, training sample size, and noise intensity. Takes one optional argument:

```bash
python sensitivity_study.py                          # run all three experiments
python sensitivity_study.py --experiment sequence_length
python sensitivity_study.py --experiment sample_size
python sensitivity_study.py --experiment noise_intensity
```

Output: `outputs/Sensitivity/`.

## 📈 Sensitivity Analysis Results

Generated by `sensitivity_study.py`; full numbers in `outputs/Sensitivity/sensitivity_summary_report.md`.

### Sequence Length

| Sequence length | Duration (s) | RMSE | Epochs |
|---|---|---|---|
| 50 | 1.0 | 2.6866 | 14 |
| 100 | 2.0 | 2.3746 | 24 |
| **250** | **5.0** | **2.3546** | 15 |
| 500 | 10.0 | 2.3628 | 21 |
| 1000 | 20.0 | 2.3797 | 10 |

**250 steps is optimal** — performance saturates and training cost keeps rising.

### Training Sample Size

| Train samples | RMSE | Epochs | Training time (s) |
|---|---|---|---|
| 500 | 2.3450 | 12 | 132 |
| 1 000 | 2.4310 | 17 | 312 |
| 2 000 | 2.3911 | 11 | 480 |
| 5 000 | 2.3891 | 29 | 2 238 |
| 10 000 | 2.3555 | 22 | 3 714 |

Going from 500 → 10 000 samples changes RMSE by only **−0.45 %**, while training cost rises ~28×.
Data efficiency is high; returns are strongly diminishing beyond moderate sample sizes.

### Noise Intensity

Observation noise (process noise held at σ = 0.1):

| σ_obs | 0.1 | 0.5 | 1.0 | 3.0 | 5.0 | 10.0 |
|---|---|---|---|---|---|---|
| Mean RMSE | 0.5269 | 0.5312 | 0.7635 | 1.5923 | 2.3820 | 4.6497 |

Process noise (observation noise held at σ = 1.0):

| σ_proc | 0.01 | 0.05 | 0.1 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|
| Mean RMSE | 0.7456 | 0.7486 | 0.7199 | 0.7554 | 0.8005 | 0.7782 |

**Key finding — a strong asymmetry.** Across the swept range, observation-noise RMSE grows by
**+782.4 %** (σ 0.1 → 10.0), while process-noise RMSE grows by only **+4.4 %** (σ 0.01 → 2.0).
KFilterNet is therefore far more sensitive to observation-side uncertainty than to process-side
uncertainty, and degrades gracefully rather than catastrophically. Note that σ ∈ [5.0, 10.0] for
observations and σ ∈ [1.0, 2.0] for process noise lie **outside** the training ranges
([0.1, 3.0] and [0.01, 0.5] respectively), so those points measure out-of-distribution behaviour.

## 📊 Model Architecture

### State Representation

- **State Vector** `xk`: `[x, y, z, vx, vy, vz, ax, ay, az]` (9-dimensional)
  - Position: `[x, y, z]` (±1000 m)
  - Velocity: `[vx, vy, vz]` (max 500 m/s)
  - Acceleration: `[ax, ay, az]` (max 12 m/s²)

- **Observation Vector** `zk`: `[x, y, z]` (3-dimensional)

### Learnable Components

Dispatch in `models/KFilterNet_single.py` (`_predict_F/H/Q/R/K`):

| Module | LEARNABLE | FIXED | SEMI_FIXED / HYBRID |
|---|---|---|---|
| Fk | Network-predicted ΔF | Constant-acceleration matrix | Fixed base + network residual |
| Hk | Network-predicted H | Fixed `I₃` with zero padding | Fixed base + network residual |
| Qk | `diag(softplus(Q_net(h)))` | `0.1·I₉` | Fixed base + network residual |
| Rk | `diag(softplus(R_net(h)))` | `0.5·I₃` | Fixed base + network residual |
| Kk | Network-predicted K | `K = PHᵀ(HPHᵀ + R + λI)⁻¹`, λ = 1e-4 | Classical solve |

Covariance propagation uses the Joseph form. Q and R are **diagonal** during tracking; only the
initial covariance `P₀` (via `CovarianceNetwork`, active when `init_type = learnable`) can be dense.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{kfilternet2025,
  title={High Adaptability Trajectory Tracking Algorithm Based on Kalman Filter Network},
  author={Li, Leyan and Zuo, Jialiang and Yang, Minbo and Yang, Rennong and Guo, Anxin and Deng, Wangchuanzi and Zhang, Zhenxing},
  journal={Acta Aeronautica et Astronautica Sinica},
  volume={},
  number={},
  year={2025},
  doi={}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Kalman filter research community for foundational work
- Trajectory tracking research community for inspiring applications

## 📞 Contact

- Project Maintainer: Leyan Li (li_leyan@163.com)

## 🔮 Future Work

- [ ] Single-target multi-sensor data fusion
- [ ] Multi-target tracking support
- [ ] Distributed tracking with communication networks
- [ ] Onboard deployment optimization
- [ ] Real-time hardware validation

---

**Note**: This project implements the algorithm described in "High Adaptability Trajectory Tracking Algorithm Based on Kalman Filter Network" published in Acta Aeronautica et Astronautica Sinica (2025). The code has been refactored for open-source publication with improved code quality, comprehensive documentation, and English language support.