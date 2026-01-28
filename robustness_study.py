"""
Robustness Test for KFilterNet and Baseline Filters

This module implements comprehensive robustness testing against:
1. Unseen noise types (Gamma, Laplace, Mixture Gaussian)
2. Unseen motion patterns (Spiral, Jerk Motion, Impulsive Turn)

The evaluation focuses on:
- Convergence behavior
- Divergence detection (NaN, Inf, state explosion)
- Performance degradation analysis
- Comparative robustness ranking
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from training.config import Config, default_config
from utils.data_generator import TrajectoryGenerator, TrajectorySample
from models.KFilterNet_single import KFilterNet_single
from models.baseline.baseline_KF import BaselineKalmanFilter
from models.baseline.baseline_AKF import BaselineAdaptiveKF
from models.baseline.baseline_IMM import BaselineIMM
from models.baseline.baseline_UKF import BaselineUKF
from models.baseline.kalman_net_tsp import KalmanNetNN


@dataclass
class DivergenceMetrics:
    """Metrics for divergence detection"""
    is_diverged: bool
    explode_state: bool
    nan_count: int
    inf_count: int
    max_state_change: float
    residual_outliers: int
    divergence_time: Optional[int] = None


@dataclass
class ConvergenceMetrics:
    """Metrics for convergence analysis"""
    initial_error: float
    middle_error: float
    final_error: float
    error_trend: str
    improvement_ratio: float
    settled_time: Optional[float]
    window_errors: List[float]


class RobustnessTest:
    """Comprehensive robustness testing for filtering algorithms"""

    def __init__(self, output_dir: str = "outputs/Robustness"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.config = default_config
        self.dt = 0.02
        self.logger = setup_logger(__name__)

        # Methods to test
        self.methods = ['KFilterNet', 'KF', 'AKF', 'IMM', 'UKF', 'KalmanNetNN']

        # Baseline filter instances
        self.filters = {
            'KF': BaselineKalmanFilter(dt=self.dt),
            'AKF': BaselineAdaptiveKF(dt=self.dt),
            'IMM': BaselineIMM(dt=self.dt),
            'UKF': BaselineUKF(dt=self.dt),
            'KalmanNetNN': None  # Will be loaded separately
        }

        # Model cache to avoid repeated loading
        self._kfilternet_model = None
        self._kalmannetnn_model = None

        # Create output subdirectories
        self.figures_dir = os.path.join(self.output_dir, "figures")
        self.reports_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def _load_kfilternet_model(self, checkpoint_path: str = None) -> KFilterNet_single:
        """Load KFilterNet model from checkpoint (cached)"""
        # Return cached model if already loaded
        if self._kfilternet_model is not None:
            return self._kfilternet_model

        if checkpoint_path is None:
            # Try to find default checkpoint in models/pretrained
            checkpoint_dir = "models/pretrained"
            if os.path.exists(checkpoint_dir):
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') or f.endswith('.pth')]
                if checkpoint_files:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                    self.logger.info(f"Using checkpoint: {checkpoint_path}")
                else:
                    self.logger.info("Warning: No checkpoint found, using untrained model")
            else:
                self.logger.info("Warning: Checkpoints directory not found, using untrained model")

        model = KFilterNet_single(self.config)
        model.eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            self.logger.info(f"Loaded model from {checkpoint_path}")

        # Cache the model
        self._kfilternet_model = model
        return model

    def _load_kalmannetnn_model(self, checkpoint_path: str = None):
        """Load KalmanNetNN model from checkpoint (cached)"""
        # Return cached model if already loaded
        if self._kalmannetnn_model is not None:
            return self._kalmannetnn_model

        if checkpoint_path is None:
            # Try to find default checkpoint in models/pretrained
            checkpoint_dir = "models/pretrained"
            if os.path.exists(checkpoint_dir):
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if 'Kalmannet' in f and (f.endswith('.pt') or f.endswith('.pth'))]
                if checkpoint_files:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                    self.logger.info(f"Using KalmanNetNN checkpoint: {checkpoint_path}")
                else:
                    self.logger.info("Warning: No KalmanNetNN checkpoint found, using untrained model")
            else:
                self.logger.info("Warning: Checkpoints directory not found, using untrained model")

        model = KalmanNetNN(self.config)
        model.eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            self.logger.info(f"Loaded KalmanNetNN model from {checkpoint_path}")

        # Cache the model
        self._kalmannetnn_model = model
        return model

    # ============================================================
    # TRAJECTORY GENERATION METHODS
    # ============================================================

    def _generate_gamma_noise_trajectory(self, duration: float = 100.0, dt: float = 0.02,
                                         shape: float = 2.0, scale: float = 0.5) -> TrajectorySample:
        """Generate trajectory with Gamma-distributed noise (asymmetric distribution)"""
        T = int(duration / dt)
        states = np.zeros((T, 9))
        observations = np.zeros((T, 3))

        # CA motion
        vx, vy, vz = 150.0, 100.0, 50.0
        x, y, z = 0.0, 0.0, 0.0
        ax, ay = 2.0, 1.0

        for t in range(T):
            states[t] = [x, y, z, vx, vy, vz, ax, ay, 0]
            x += vx * dt + 0.5 * ax * dt**2
            y += vy * dt + 0.5 * ay * dt**2
            z += vz * dt
            vx += ax * dt
            vy += ay * dt

            # Generate Gamma noise and center it
            noise = np.random.gamma(shape, scale, 3)
            noise = noise - noise.mean()  # Center to zero mean
            observations[t] = states[t, 0:3] + noise

        return TrajectorySample(states=states, observations=observations,
                                process_noise=0.1, observation_noise=scale**2 * shape,
                                motion_type='ca_with_gamma_noise', dt=dt,
                                metadata={'noise_type': 'gamma', 'shape': shape, 'scale': scale})

    def _generate_laplace_noise_trajectory(self, duration: float = 100.0, dt: float = 0.02,
                                           loc: float = 0.0, scale: float = 0.5) -> TrajectorySample:
        """Generate trajectory with Laplace-distributed noise (heavy-tailed)"""
        T = int(duration / dt)
        states = np.zeros((T, 9))
        observations = np.zeros((T, 3))

        # CA motion
        vx, vy, vz = 150.0, 100.0, 50.0
        x, y, z = 0.0, 0.0, 0.0
        ax, ay = 2.0, 1.0

        for t in range(T):
            states[t] = [x, y, z, vx, vy, vz, ax, ay, 0]
            x += vx * dt + 0.5 * ax * dt**2
            y += vy * dt + 0.5 * ay * dt**2
            z += vz * dt
            vx += ax * dt
            vy += ay * dt

            noise = np.random.laplace(loc, scale, 3)
            observations[t] = states[t, 0:3] + noise

        return TrajectorySample(states=states, observations=observations,
                                process_noise=0.1, observation_noise=2 * scale**2,
                                motion_type='ca_with_laplace_noise', dt=dt,
                                metadata={'noise_type': 'laplace', 'loc': loc, 'scale': scale})

    def _generate_mixture_gaussian_trajectory(self, duration: float = 100.0, dt: float = 0.02) -> TrajectorySample:
        """Generate trajectory with Gaussian mixture noise"""
        T = int(duration / dt)
        states = np.zeros((T, 9))
        observations = np.zeros((T, 3))

        # CA motion
        vx, vy, vz = 150.0, 100.0, 50.0
        x, y, z = 0.0, 0.0, 0.0
        ax, ay = 2.0, 1.0

        # Mixture components: 70% from N(0, 0.5), 30% from N(0, 2.0)
        for t in range(T):
            states[t] = [x, y, z, vx, vy, vz, ax, ay, 0]
            x += vx * dt + 0.5 * ax * dt**2
            y += vy * dt + 0.5 * ay * dt**2
            z += vz * dt
            vx += ax * dt
            vy += ay * dt

            noise = np.zeros(3)
            for i in range(3):
                if np.random.random() < 0.7:
                    noise[i] = np.random.normal(0, 0.5)
                else:
                    noise[i] = np.random.normal(0, 2.0)

            observations[t] = states[t, 0:3] + noise

        return TrajectorySample(states=states, observations=observations,
                                process_noise=0.1, observation_noise=1.65,
                                motion_type='ca_with_mixture_gaussian', dt=dt,
                                metadata={'noise_type': 'mixture_gaussian'})

    def _generate_spiral_maneuver_trajectory(self, duration: float = 100.0, dt: float = 0.02,
                                             omega: float = 0.1, v_r: float = 10.0,
                                             v_z: float = 20.0) -> TrajectorySample:
        """Generate 3D spiral maneuver trajectory (unseen motion pattern)"""
        T = int(duration / dt)
        states = np.zeros((T, 9))
        observations = np.zeros((T, 3))

        R0 = 100.0  # Initial radius
        x0, y0, z0 = R0, 0.0, 0.0

        x, y, z = x0, y0, z0
        R = R0

        for t in range(T):
            theta = omega * t * dt
            R_current = R0 + v_r * t * dt
            v_theta = R_current * omega

            # Position
            x_new = R_current * np.cos(theta)
            y_new = R_current * np.sin(theta)
            z_new = v_z * t * dt

            # Velocity
            vx = -v_theta * np.sin(theta) + v_r * np.cos(theta)
            vy = v_theta * np.cos(theta) + v_r * np.sin(theta)
            vz = v_z

            # Acceleration (centripetal + radial change)
            ax = -omega**2 * R_current * np.cos(theta) - 2 * v_r * omega * np.sin(theta)
            ay = -omega**2 * R_current * np.sin(theta) + 2 * v_r * omega * np.cos(theta)
            az = 0.0

            states[t] = [x_new, y_new, z_new, vx, vy, vz, ax, ay, 0]
            x, y, z = x_new, y_new, z_new

            # Add Gaussian noise
            observations[t] = states[t, 0:3] + np.random.normal(0, 0.5, 3)

        return TrajectorySample(states=states, observations=observations,
                                process_noise=0.1, observation_noise=0.25,
                                motion_type='spiral_maneuver', dt=dt,
                                metadata={'omega': omega, 'v_r': v_r, 'v_z': v_z})

    def _generate_jerk_motion_trajectory(self, duration: float = 100.0, dt: float = 0.02,
                                         jerk: float = 0.5) -> TrajectorySample:
        """Generate constant jerk motion (acceleration changes linearly)"""
        T = int(duration / dt)
        states = np.zeros((T, 9))
        observations = np.zeros((T, 3))

        x, y, z = 0.0, 0.0, 0.0
        vx, vy, vz = 150.0, 100.0, 50.0
        ax, ay = 0.0, 0.0

        for t in range(T):
            states[t] = [x, y, z, vx, vy, vz, ax, ay, jerk]

            # Position update (with jerk term)
            x += vx * dt + 0.5 * ax * dt**2 + (1/6) * jerk * dt**3
            y += vy * dt + 0.5 * ay * dt**2 + (1/6) * jerk * dt**3
            z += vz * dt

            # Velocity update
            vx += ax * dt + 0.5 * jerk * dt**2
            vy += ay * dt + 0.5 * jerk * dt**2

            # Acceleration update
            ax += jerk * dt
            ay += jerk * dt

            # Add Gaussian noise
            observations[t] = states[t, 0:3] + np.random.normal(0, 0.5, 3)

        return TrajectorySample(states=states, observations=observations,
                                process_noise=0.1, observation_noise=0.25,
                                motion_type='jerk_motion', dt=dt,
                                metadata={'jerk': jerk})

    def _generate_impulsive_turn_trajectory(self, duration: float = 100.0, dt: float = 0.02,
                                            impulse_time: float = 30.0, turn_rate: float = 0.5) -> TrajectorySample:
        """Generate trajectory with impulsive turn (turn rate sudden change)"""
        T = int(duration / dt)
        states = np.zeros((T, 9))
        observations = np.zeros((T, 3))

        x, y, z = 0.0, 0.0, 0.0
        vx, vy, vz = 150.0, 100.0, 50.0
        ax, ay = 0.0, 0.0

        impulse_idx = int(impulse_time / dt)
        current_omega = 0.0

        for t in range(T):
            states[t] = [x, y, z, vx, vy, vz, ax, ay, 0]

            if t >= impulse_idx:
                current_omega = turn_rate
                # Apply turn
                sin_wt = np.sin(current_omega * dt)
                cos_wt = np.cos(current_omega * dt)

                new_vx = cos_wt * vx + sin_wt * vy
                new_vy = -sin_wt * vx + cos_wt * vy

                vx, vy = new_vx, new_vy

                # Centripetal acceleration
                v_mag = np.sqrt(vx**2 + vy**2)
                ax = -current_omega * vy
                ay = current_omega * vx
            else:
                ax, ay = 0.0, 0.0

            # Position update
            x += vx * dt + 0.5 * ax * dt**2
            y += vy * dt + 0.5 * ay * dt**2
            z += vz * dt

            # Velocity update
            vx += ax * dt
            vy += ay * dt

            # Add Gaussian noise
            observations[t] = states[t, 0:3] + np.random.normal(0, 0.5, 3)

        return TrajectorySample(states=states, observations=observations,
                                process_noise=0.1, observation_noise=0.25,
                                motion_type='impulsive_turn', dt=dt,
                                metadata={'impulse_time': impulse_time, 'turn_rate': turn_rate})

    # ============================================================
    # EVALUATION METHODS
    # ============================================================

    def evaluate_method(self, method: str, observations: np.ndarray,
                        true_states: np.ndarray) -> Dict:
        """Evaluate a single filtering method on given observations"""
        if method == 'KFilterNet':
            # Load KFilterNet model
            model = self._load_kfilternet_model()
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observations).unsqueeze(0)
                pred_states, _, _ = model(obs_tensor)
                pred_states = pred_states.squeeze(0).cpu().numpy()
        elif method in self.filters and self.filters[method] is not None:
            pred_states = self.filters[method].filter(observations)
        elif method == 'KalmanNetNN':
            # Load KalmanNetNN model
            model = self._load_kalmannetnn_model()
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observations).unsqueeze(0)
                pred_states, _, _ = model(obs_tensor)
                pred_states = pred_states.squeeze(0).cpu().numpy()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate RMSE
        position_errors = pred_states[:, 0:3] - true_states[:, 0:3]
        rmse_total = np.sqrt(np.mean(position_errors**2))
        rmse_x = np.sqrt(np.mean(position_errors[:, 0]**2))
        rmse_y = np.sqrt(np.mean(position_errors[:, 1]**2))
        rmse_z = np.sqrt(np.mean(position_errors[:, 2]**2))

        rmse_per_timestep = np.sqrt(np.sum(position_errors**2, axis=1))

        return {
            'method': method,
            'rmse_total': rmse_total,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'rmse_z': rmse_z,
            'rmse_per_timestep': rmse_per_timestep,
            'predicted_states': pred_states
        }

    # ============================================================
    # DIVERGENCE DETECTION AND CONVERGENCE ANALYSIS
    # ============================================================

    def _check_divergence(self, predicted_states: np.ndarray,
                          observations: np.ndarray) -> DivergenceMetrics:
        """Check if the filter has diverged"""
        T = len(predicted_states)

        metrics = DivergenceMetrics(
            is_diverged=False,
            explode_state=False,
            nan_count=0,
            inf_count=0,
            max_state_change=0.0,
            residual_outliers=0,
            divergence_time=None
        )

        # Check NaN and Inf
        metrics.nan_count = int(np.sum(np.isnan(predicted_states)))
        metrics.inf_count = int(np.sum(np.isinf(predicted_states)))

        if metrics.nan_count > 0 or metrics.inf_count > 0:
            metrics.is_diverged = True
            # Find first divergence time
            for t in range(T):
                if np.any(np.isnan(predicted_states[t])) or np.any(np.isinf(predicted_states[t])):
                    metrics.divergence_time = t
                    break
            return metrics

        # Check state explosion
        state_changes = np.diff(predicted_states, axis=0)
        metrics.max_state_change = float(np.max(np.abs(state_changes)))

        if metrics.max_state_change > 1000.0:
            metrics.explode_state = True
            metrics.is_diverged = True
            # Find explosion time
            for t in range(1, T):
                if np.max(np.abs(state_changes[t-1])) > 1000.0:
                    metrics.divergence_time = t
                    break

        # Check residual outliers
        residuals = observations - predicted_states[:, 0:3]
        residual_mags = np.linalg.norm(residuals, axis=1)
        median = np.median(residual_mags)
        mad = np.median(np.abs(residual_mags - median))
        if mad > 0:
            threshold = median + 10 * mad
        else:
            threshold = median + 10

        metrics.residual_outliers = int(np.sum(residual_mags > threshold))

        return metrics

    def _analyze_convergence(self, predicted_states: np.ndarray,
                             true_states: np.ndarray) -> ConvergenceMetrics:
        """Analyze convergence behavior"""
        T = len(predicted_states)

        # Calculate errors at each timestep
        errors = np.linalg.norm(predicted_states[:, 0:3] - true_states[:, 0:3], axis=1)

        # Window-based analysis
        window_size = max(50, T // 20)
        window_errors = []

        for i in range(0, T, window_size):
            end_idx = min(i + window_size, T)
            if end_idx > i:
                window_errors.append(float(np.mean(errors[i:end_idx])))

        initial_error = window_errors[0] if window_errors else float(np.mean(errors))
        middle_idx = len(window_errors) // 2
        middle_error = window_errors[middle_idx] if window_errors else initial_error
        final_error = window_errors[-1] if window_errors else float(np.mean(errors))

        error_trend = 'decreasing' if final_error < initial_error else 'increasing'

        if initial_error > 1e-10:
            improvement_ratio = (initial_error - final_error) / initial_error
        else:
            improvement_ratio = 0.0

        # Find settled time
        settled_time = None
        if len(window_errors) > 1:
            threshold = 0.01 * initial_error
            for i in range(1, len(window_errors)):
                if abs(window_errors[i] - window_errors[i-1]) < threshold:
                    settled_time = i * window_size * self.dt
                    break

        return ConvergenceMetrics(
            initial_error=initial_error,
            middle_error=middle_error,
            final_error=final_error,
            error_trend=error_trend,
            improvement_ratio=improvement_ratio,
            settled_time=settled_time,
            window_errors=window_errors
        )

    # ============================================================
    # EXPERIMENT 1: UNSEEN NOISE TYPES
    # ============================================================

    def experiment_1_unseen_noise(self, num_samples: int = 50):
        """Experiment 1: Robustness test against unseen noise types"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("Robustness Test 1: Unseen Noise Types")
        self.logger.info("=" * 70)

        noise_configs = [
            {
                'name': 'gamma',
                'description': 'Gamma Distribution (Asymmetric)',
                'func': self._generate_gamma_noise_trajectory,
                'params': {'shape': 2.0, 'scale': 0.5}
            },
            {
                'name': 'laplace',
                'description': 'Laplace Distribution (Heavy-tailed)',
                'func': self._generate_laplace_noise_trajectory,
                'params': {'loc': 0.0, 'scale': 0.5}
            },
            {
                'name': 'mixture_gauss',
                'description': 'Mixture Gaussian',
                'func': self._generate_mixture_gaussian_trajectory,
                'params': None
            }
        ]

        results = {}

        for noise_config in noise_configs:
            self.logger.info(f"\nTesting with {noise_config['description']}...")
            config_results = []

            # Generate test samples
            test_samples = []
            for _ in range(num_samples):
                if noise_config['params']:
                    sample = noise_config['func'](**noise_config['params'])
                else:
                    sample = noise_config['func']()
                test_samples.append(sample)

            # Evaluate each method
            for method in self.methods:
                self.logger.info(f"  Evaluating {method}...")
                method_stats = {
                    'method': method,
                    'rmse_values': [],
                    'divergence_count': 0,
                    'nan_count': 0,
                    'inf_count': 0,
                    'convergence_metrics': [],
                    'divergence_times': []
                }

                for sample in test_samples:
                    result = self.evaluate_method(method, sample.observations, sample.states)
                    divergence = self._check_divergence(result['predicted_states'], sample.observations)
                    convergence = self._analyze_convergence(result['predicted_states'], sample.states)

                    method_stats['rmse_values'].append(result['rmse_total'])
                    method_stats['divergence_count'] += 1 if divergence.is_diverged else 0
                    method_stats['nan_count'] += divergence.nan_count
                    method_stats['inf_count'] += divergence.inf_count
                    method_stats['convergence_metrics'].append(convergence)
                    if divergence.divergence_time is not None:
                        method_stats['divergence_times'].append(divergence.divergence_time)

                # Summary statistics
                config_results.append({
                    'method': method,
                    'avg_rmse': float(np.mean(method_stats['rmse_values'])),
                    'std_rmse': float(np.std(method_stats['rmse_values'])),
                    'min_rmse': float(np.min(method_stats['rmse_values'])),
                    'max_rmse': float(np.max(method_stats['rmse_values'])),
                    'divergence_rate': method_stats['divergence_count'] / len(test_samples),
                    'nan_per_million': (method_stats['nan_count'] / (len(test_samples) * len(sample.states) * 9)) * 1e6,
                    'avg_improvement': float(np.mean([c.improvement_ratio for c in method_stats['convergence_metrics']])),
                    'avg_settled_time': float(np.mean([c.settled_time for c in method_stats['convergence_metrics']
                                                       if c.settled_time is not None])),
                    'avg_divergence_time': float(np.mean(method_stats['divergence_times']))
                    if method_stats['divergence_times'] else None
                })

            results[noise_config['name']] = {
                'description': noise_config['description'],
                'results': config_results
            }

            self.logger.info(f"    Completed {noise_config['name']} test")

        # Save results
        self._save_experiment_1_results(results)
        self._visualize_experiment_1_results(results)

        return results

    # ============================================================
    # EXPERIMENT 2: UNSEEN MOTION PATTERNS
    # ============================================================

    def experiment_2_unseen_motion(self, num_samples: int = 50):
        """Experiment 2: Robustness test against unseen motion patterns"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("Robustness Test 2: Unseen Motion Patterns")
        self.logger.info("=" * 70)

        motion_configs = [
            {
                'name': 'spiral',
                'description': '3D Spiral Maneuver',
                'func': self._generate_spiral_maneuver_trajectory,
                'params': {'omega': 0.1, 'v_r': 10.0, 'v_z': 20.0}
            },
            {
                'name': 'jerk_motion',
                'description': 'Constant Jerk Motion',
                'func': self._generate_jerk_motion_trajectory,
                'params': {'jerk': 0.5}
            },
            {
                'name': 'impulsive_turn',
                'description': 'Impulsive Turn (Abrupt Turn Rate Change)',
                'func': self._generate_impulsive_turn_trajectory,
                'params': {'impulse_time': 30.0, 'turn_rate': 0.5}
            }
        ]

        results = {}

        for motion_config in motion_configs:
            self.logger.info(f"\nTesting with {motion_config['description']}...")
            config_results = []

            # Generate test samples
            test_samples = []
            for _ in range(num_samples):
                if motion_config['params']:
                    sample = motion_config['func'](**motion_config['params'])
                else:
                    sample = motion_config['func']()
                test_samples.append(sample)

            # Evaluate each method
            for method in self.methods:
                self.logger.info(f"  Evaluating {method}...")
                method_stats = {
                    'method': method,
                    'rmse_values': [],
                    'divergence_count': 0,
                    'nan_count': 0,
                    'inf_count': 0,
                    'convergence_metrics': [],
                    'divergence_times': []
                }

                for sample in test_samples:
                    result = self.evaluate_method(method, sample.observations, sample.states)
                    divergence = self._check_divergence(result['predicted_states'], sample.observations)
                    convergence = self._analyze_convergence(result['predicted_states'], sample.states)

                    method_stats['rmse_values'].append(result['rmse_total'])
                    method_stats['divergence_count'] += 1 if divergence.is_diverged else 0
                    method_stats['nan_count'] += divergence.nan_count
                    method_stats['inf_count'] += divergence.inf_count
                    method_stats['convergence_metrics'].append(convergence)
                    if divergence.divergence_time is not None:
                        method_stats['divergence_times'].append(divergence.divergence_time)

                # Summary statistics
                config_results.append({
                    'method': method,
                    'avg_rmse': float(np.mean(method_stats['rmse_values'])),
                    'std_rmse': float(np.std(method_stats['rmse_values'])),
                    'min_rmse': float(np.min(method_stats['rmse_values'])),
                    'max_rmse': float(np.max(method_stats['rmse_values'])),
                    'divergence_rate': method_stats['divergence_count'] / len(test_samples),
                    'nan_per_million': (method_stats['nan_count'] / (len(test_samples) * len(sample.states) * 9)) * 1e6,
                    'avg_improvement': float(np.mean([c.improvement_ratio for c in method_stats['convergence_metrics']])),
                    'avg_settled_time': float(np.mean([c.settled_time for c in method_stats['convergence_metrics']
                                                       if c.settled_time is not None])),
                    'avg_divergence_time': float(np.mean(method_stats['divergence_times']))
                    if method_stats['divergence_times'] else None
                })

            results[motion_config['name']] = {
                'description': motion_config['description'],
                'results': config_results
            }

            self.logger.info(f"    Completed {motion_config['name']} test")

        # Save results
        self._save_experiment_2_results(results)
        self._visualize_experiment_2_results(results)

        return results

    # ============================================================
    # SAVE AND VISUALIZATION METHODS
    # ============================================================

    def _save_experiment_1_results(self, results: Dict):
        """Save Experiment 1 results"""
        output_path = os.path.join(self.output_dir, "experiment_1_unseen_noise.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Experiment 1 results saved to: {output_path}")

    def _save_experiment_2_results(self, results: Dict):
        """Save Experiment 2 results"""
        output_path = os.path.join(self.output_dir, "experiment_2_unseen_motion.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Experiment 2 results saved to: {output_path}")

    def _visualize_experiment_1_results(self, results: Dict):
        """Visualize Experiment 1 results"""
        noise_names = list(results.keys())
        methods = self.methods

        # Extract data for heatmap
        rmse_matrix = np.zeros((len(methods), len(noise_names)))
        divergence_matrix = np.zeros((len(methods), len(noise_names)))

        for i, method in enumerate(methods):
            for j, noise_name in enumerate(noise_names):
                method_result = results[noise_name]['results'][i]
                rmse_matrix[i, j] = method_result['avg_rmse']
                divergence_matrix[i, j] = method_result['divergence_rate']

        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Plot 1: RMSE Heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(rmse_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax1.set_xticks(range(len(noise_names)))
        ax1.set_xticklabels([results[n]['description'] for n in noise_names], rotation=45, ha='right', fontsize=9)
        ax1.set_yticks(range(len(methods)))
        ax1.set_yticklabels(methods, fontsize=11)
        ax1.set_title('RMSE by Noise Type', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='RMSE (m)')

        # Add values to heatmap
        for i in range(len(methods)):
            for j in range(len(noise_names)):
                ax1.text(j, i, f'{rmse_matrix[i, j]:.3f}', ha='center', va='center',
                        color='black' if rmse_matrix[i, j] < np.max(rmse_matrix) * 0.7 else 'white',
                        fontsize=8)

        # Plot 2: Divergence Rate Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(divergence_matrix, aspect='auto', cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
        ax2.set_xticks(range(len(noise_names)))
        ax2.set_xticklabels([results[n]['description'] for n in noise_names], rotation=45, ha='right', fontsize=9)
        ax2.set_yticks(range(len(methods)))
        ax2.set_yticklabels(methods, fontsize=11)
        ax2.set_title('Divergence Rate by Noise Type', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='Divergence Rate')

        # Add values to divergence heatmap
        for i in range(len(methods)):
            for j in range(len(noise_names)):
                val = divergence_matrix[i, j]
                ax2.text(j, i, f'{val:.2f}' if val > 0 else '0', ha='center', va='center',
                        color='black' if val < 0.5 else 'white', fontsize=10)

        # Plot 3: RMSE Bar Chart Comparison
        ax3 = fig.add_subplot(gs[1, :])
        x = np.arange(len(noise_names))
        width = 0.12

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, method in enumerate(methods):
            values = [results[n]['results'][i]['avg_rmse'] for n in noise_names]
            bars = ax3.bar(x + i * width, values, width, label=method, color=colors[i], alpha=0.8, edgecolor='k', linewidth=0.5)

        ax3.set_xlabel('Noise Type', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Average RMSE (m)', fontsize=11, fontweight='bold')
        ax3.set_title('RMSE Comparison Across Noise Types', fontsize=12, fontweight='bold')
        ax3.set_xticks(x + width * (len(methods) - 1) / 2)
        ax3.set_xticklabels([results[n]['description'] for n in noise_names], rotation=45, ha='right')
        ax3.legend(fontsize=9, ncol=3, loc='upper right')
        ax3.grid(True, alpha=0.3, axis='y')

        # Save figure
        fig_path = os.path.join(self.figures_dir, "experiment_1_unseen_noise_robustness.png")
        plt.savefig(fig_path, dpi=400, bbox_inches='tight', facecolor='white')
        plt.close()
        self.logger.info(f"Experiment 1 visualization saved to: {fig_path}")

    def _visualize_experiment_2_results(self, results: Dict):
        """Visualize Experiment 2 results"""
        motion_names = list(results.keys())
        methods = self.methods

        # Extract data for heatmap
        rmse_matrix = np.zeros((len(methods), len(motion_names)))
        divergence_matrix = np.zeros((len(methods), len(motion_names)))

        for i, method in enumerate(methods):
            for j, motion_name in enumerate(motion_names):
                method_result = results[motion_name]['results'][i]
                rmse_matrix[i, j] = method_result['avg_rmse']
                divergence_matrix[i, j] = method_result['divergence_rate']

        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Plot 1: RMSE Heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(rmse_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax1.set_xticks(range(len(motion_names)))
        ax1.set_xticklabels([results[m]['description'] for m in motion_names], rotation=45, ha='right', fontsize=9)
        ax1.set_yticks(range(len(methods)))
        ax1.set_yticklabels(methods, fontsize=11)
        ax1.set_title('RMSE by Motion Type', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='RMSE (m)')

        # Add values to heatmap
        for i in range(len(methods)):
            for j in range(len(motion_names)):
                ax1.text(j, i, f'{rmse_matrix[i, j]:.3f}', ha='center', va='center',
                        color='black' if rmse_matrix[i, j] < np.max(rmse_matrix) * 0.7 else 'white',
                        fontsize=8)

        # Plot 2: Divergence Rate Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(divergence_matrix, aspect='auto', cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
        ax2.set_xticks(range(len(motion_names)))
        ax2.set_xticklabels([results[m]['description'] for m in motion_names], rotation=45, ha='right', fontsize=9)
        ax2.set_yticks(range(len(methods)))
        ax2.set_yticklabels(methods, fontsize=11)
        ax2.set_title('Divergence Rate by Motion Type', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='Divergence Rate')

        # Add values to divergence heatmap
        for i in range(len(methods)):
            for j in range(len(motion_names)):
                val = divergence_matrix[i, j]
                ax2.text(j, i, f'{val:.2f}' if val > 0 else '0', ha='center', va='center',
                        color='black' if val < 0.5 else 'white', fontsize=10)

        # Plot 3: RMSE Bar Chart Comparison
        ax3 = fig.add_subplot(gs[1, :])
        x = np.arange(len(motion_names))
        width = 0.12

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, method in enumerate(methods):
            values = [results[m]['results'][i]['avg_rmse'] for m in motion_names]
            bars = ax3.bar(x + i * width, values, width, label=method, color=colors[i], alpha=0.8, edgecolor='k', linewidth=0.5)

        ax3.set_xlabel('Motion Type', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Average RMSE (m)', fontsize=11, fontweight='bold')
        ax3.set_title('RMSE Comparison Across Motion Types', fontsize=12, fontweight='bold')
        ax3.set_xticks(x + width * (len(methods) - 1) / 2)
        ax3.set_xticklabels([results[m]['description'] for m in motion_names], rotation=45, ha='right')
        ax3.legend(fontsize=9, ncol=3, loc='upper right')
        ax3.grid(True, alpha=0.3, axis='y')

        # Save figure
        fig_path = os.path.join(self.figures_dir, "experiment_2_unseen_motion_robustness.png")
        plt.savefig(fig_path, dpi=400, bbox_inches='tight', facecolor='white')
        plt.close()
        self.logger.info(f"Experiment 2 visualization saved to: {fig_path}")

    def _generate_robustness_report(self, results1: Dict, results2: Dict):
        """Generate comprehensive robustness report"""
        report_path = os.path.join(self.reports_dir, "robustness_report.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# KFilterNet Robustness Test Report\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")
            f.write("This report presents the results of comprehensive robustness testing ")
            f.write("of KFilterNet and baseline filtering algorithms against:\n\n")
            f.write("1. **Unseen Noise Types**: Gamma, Laplace, Mixture Gaussian\n")
            f.write("2. **Unseen Motion Patterns**: Spiral, Jerk Motion, Impulsive Turn\n\n")

            # Experiment 1 Summary
            f.write("## Experiment 1: Unseen Noise Types\n\n")
            f.write("### RMSE Summary\n\n")
            f.write("| Noise Type | KFilterNet | KF | AKF | IMM | UKF | Best |\n")
            f.write("|------------|------------|----|----|----|----|----|\n")

            for noise_name in results1:
                r = results1[noise_name]['results']
                vals = [res['avg_rmse'] for res in r]
                best_idx = int(np.argmin(vals))
                best_method = self.methods[best_idx]
                f.write(f"| {results1[noise_name]['description']} |")
                f.write(f" {r[0]['avg_rmse']:.4f} | {r[1]['avg_rmse']:.4f} | {r[2]['avg_rmse']:.4f} |")
                f.write(f" {r[3]['avg_rmse']:.4f} | {r[4]['avg_rmse']:.4f} | {best_method} |\n")

            f.write("\n### Divergence Rate Summary\n\n")
            f.write("| Noise Type | KFilterNet | KF | AKF | IMM | UKF |\n")
            f.write("|------------|------------|----|----|----|----|\n")

            for noise_name in results1:
                r = results1[noise_name]['results']
                f.write(f"| {results1[noise_name]['description']} |")
                for i in range(5):  # Exclude KalmanNetNN for simplicity
                    f.write(f" {r[i]['divergence_rate']:.2%} |")
                f.write("\n")

            # Experiment 2 Summary
            f.write("\n## Experiment 2: Unseen Motion Patterns\n\n")
            f.write("### RMSE Summary\n\n")
            f.write("| Motion Type | KFilterNet | KF | AKF | IMM | UKF | Best |\n")
            f.write("|-------------|------------|----|----|----|----|----|\n")

            for motion_name in results2:
                r = results2[motion_name]['results']
                vals = [res['avg_rmse'] for res in r]
                best_idx = int(np.argmin(vals))
                best_method = self.methods[best_idx]
                f.write(f"| {results2[motion_name]['description']} |")
                f.write(f" {r[0]['avg_rmse']:.4f} | {r[1]['avg_rmse']:.4f} | {r[2]['avg_rmse']:.4f} |")
                f.write(f" {r[3]['avg_rmse']:.4f} | {r[4]['avg_rmse']:.4f} | {best_method} |\n")

            f.write("\n### Divergence Rate Summary\n\n")
            f.write("| Motion Type | KFilterNet | KF | AKF | IMM | UKF |\n")
            f.write("|-------------|------------|----|----|----|----|\n")

            for motion_name in results2:
                r = results2[motion_name]['results']
                f.write(f"| {results2[motion_name]['description']} |")
                for i in range(5):
                    f.write(f" {r[i]['divergence_rate']:.2%} |")
                f.write("\n")

            # Conclusions
            f.write("\n## Conclusions\n\n")
            f.write("### Key Findings\n\n")
            f.write("1. **Noise Robustness**: ")
            best_noise = np.mean([results1[n]['results'][0]['avg_rmse'] for n in results1])
            f.write(f"KFilterNet achieves average RMSE of {best_noise:.4f} across unseen noise types.\n\n")

            f.write("2. **Motion Robustness**: ")
            best_motion = np.mean([results2[m]['results'][0]['avg_rmse'] for m in results2])
            f.write(f"KFilterNet achieves average RMSE of {best_motion:.4f} across unseen motion patterns.\n\n")

            f.write("### Robustness Ranking\n\n")
            f.write("Overall robustness ranking (lower divergence rate, lower RMSE is better):\n\n")
            for method in self.methods:
                avg_rmse = (np.mean([results1[n]['results'][i]['avg_rmse'] for n in results1 for i in range(len(self.methods)) if self.methods[i] == method]) +
                           np.mean([results2[m]['results'][i]['avg_rmse'] for m in results2 for i in range(len(self.methods)) if self.methods[i] == method])) / 2
                f.write(f"- **{method}**: Average RMSE ≈ {avg_rmse:.4f}\n")

        self.logger.info(f"Robustness report saved to: {report_path}")

    # ============================================================
    # MAIN RUN METHODS
    # ============================================================

    def run_all_experiments(self, num_samples: int = 50):
        """Run all robustness experiments"""
        self.logger.info("=" * 70)
        self.logger.info("Starting Comprehensive Robustness Testing")
        self.logger.info("=" * 70)

        # Run experiments
        results1 = self.experiment_1_unseen_noise(num_samples=num_samples)
        results2 = self.experiment_2_unseen_motion(num_samples=num_samples)

        # Generate comprehensive report
        self._generate_robustness_report(results1, results2)

        self.logger.info("\n" + "=" * 70)
        self.logger.info("All Robustness Experiments Completed!")
        self.logger.info("=" * 70)
        self.logger.info(f"Results saved to: {self.output_dir}")

        return results1, results2


def main():
    """Main function"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create robustness test instance
    robustness = RobustnessTest()

    # Run all experiments
    results1, results2 = robustness.run_all_experiments(num_samples=50)

    logger.info("\nRobustness testing completed!")


if __name__ == "__main__":
    main()
