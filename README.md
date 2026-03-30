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
Utilizes **Gated Recurrent Units (GRUs)** to extract features from historical measurement sequences `Z0:k = [z0, z1, ..., zk]`, enabling the model to learn temporal dependencies and adapt to changing conditions.

### 3. Flexible Configuration Modes
Supports three module configuration types for different operational requirements:
- **FIXED**: classical matrix operations
- **SEMI_FIXED**: base matrix + neural residual learning
- **LEARNABLE**: fully neural network-based prediction

## 🚀 Features

- **Flexible Module Configurations**: Each of the 5 Kalman filter components (Fk, Hk, Kk, Qk, Rk) can be independently configured as:
  - Learnable (fully neural network-based)
  - Fixed (classical matrix operations)
  - Semi-fixed (base + residual learning)

- **Comprehensive Motion Models**:
  - Constant Velocity (CV)
  - Constant Acceleration (CA)
  - Coordinated Turn (CT)
  - Maneuvering motion (including weaving, climbing, diving)

- **Advanced Noise Models**:
  - Gaussian noise with time-varying statistics
  - Impulse noise (impulsion with 5% probability)
  - Correlated noise between timesteps

- **Robust Training**:
  - Numerical stability enforcement (eigenvalue clamping)
  - Early stopping and learning rate scheduling
  - Gradient clipping
  - Checkpoint management

## 📁 Project Structure

```
KFilterNet-single/
├── models/                     # Model implementations
│   ├── KFilterNet_single.py    # Main KFilterNet model
│   ├── kalman_net_tsp.py       # KalmanNet baseline
│   ├── components/            # Neural network components
│   │   ├── kalman_components.py
│   │   ├── mlp.py
│   │   ├── noise_network.py
│   │   ├── history_encoder.py
│   │   └── covariance_network.py
│   └── baseline/              # Baseline filter implementations
│       ├── baseline_KF.py
│       ├── baseline_AKF.py
│       ├── baseline_IMM.py
│       └── baseline_UKF.py
├── training/                   # Training utilities
│   ├── config.py              # Configuration management
│   ├── trainer.py             # Training loop
│   └── data_loader.py         # Data loading
├── utils/                      # Utility functions
│   ├── data_generator.py      # Trajectory generation
│   ├── visualization.py       # Visualization tools
│   └── logger.py              # Logging setup
├── constants.py                # Centralized constants
├── main.py                     # Main training script
├── ablation_study.py          # Ablation study (7 motion modes)
├── comparison_study.py        # Comparison with baselines
├── robustness_study.py        # Robustness analysis
└── requirements.txt            # Dependencies
```

## 🚀 Quick Start

### Basic Training

Train the default KFilterNet model:
```bash
python main.py
```

### Custom Training with All Modules Learnable

```bash
python main.py \
    --model_type KFilterNet_single \
    --H_type learnable \
    --K_type learnable \
    --F_type learnable \
    --Q_type learnable \
    --R_type learnable \
    --epochs 150 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --hidden_dim 128 \
    --device cpu
```

### Ablation Study

Run comprehensive ablation studies across 7 motion modes:
```bash
python ablation_study.py
```

This will evaluate different module configurations:
- K - Learnable, Q/R - Fixed, F/H - Semi-Fixed
- K - Fixed, Q/R - Fixed, F/H - Learnable
- And 5 more configurations...

### Comparison with Baselines

Compare against KF, AKF, IMM, UKF, and KalmanNet:
```bash
python comparison_study.py
```

### Robustness Analysis

Evaluate performance under:
- Different noise levels (Proc. Noise, Obs. Noise variations)
- Complex noise environments (Gaussian, Impulse, Correlated)
- Trajectory robustness analysis

```bash
python robustness_study.py
```

## 📊 Model Architecture

### State Representation

- **State Vector** `xk`: `[x, y, z, vx, vy, vz, ax, ay, az]` (9-dimensional)
  - Position: `[x, y, z]`
  - Velocity: `[vx, vy, vz]` (max 500 m/s)
  - Acceleration: `[ax, ay, az]` (max 12 m/s²)

- **Observation Vector** `zk`: `[x, y, z]` (3-dimensional)

### Module Types

#### FIXED Mode
Uses classical matrix operations:
```
Fk = F_CA (constant kinematic matrix)
Hk = I3 with zero padding
Qk, Rk = fixed covariance matrices
Kk = computed via standard Kalman equation
```

#### SEMI_FIXED Mode
Base matrix + neural residual:
```
Fk = F_CA + F_theta(ck; theta_F)
Hk = H_fixed + H_theta(ck; theta_H)
Qk = Q* + Q_theta(ck; theta_Q)
```

#### LEARNABLE Mode
Fully neural network-based:
```
Fk = F_theta(ck; theta_F)
Hk = H_theta(ck; theta_H)
Qk = diag(softplus(Q_theta(ck; theta_Q)))
Kk = K_theta(ck; theta_K)
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{kfilternet2025,
  title={High Adaptability Trajectory Tracking Algorithm Based on Kalman Filter Network},
  author={Li, Leyan and Zuo, Jialiang and Yang, Minbo and Yang, Rennong and Guo, Anxin and Deng, Wangchuanzi and Zhang, Zhenxing},
  year={2025}
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
