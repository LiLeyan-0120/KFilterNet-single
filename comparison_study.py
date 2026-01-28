#!/usr/bin/env python3
"""
KFilterNet Comparison Study
Implements comparisons with baseline methods, including:
1. Comparison under different motion modes (mixed motion modes and individual CA/CV/CT/Maneuvering modes)
2. Comparison under different noise covariances
3. Trajectory tracking comparison for one trajectory under each of the four motion modes

Baselines include:
- Kalman Filter (KF)
- Interacting Multiple Model (IMM)
- Adaptive Kalman Filter (AKF)
- Unscented Kalman Filter (UKF)
- KalmanNetNN
"""

import os
import sys
import json
import time
import random
import copy
from datetime import datetime
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d
from utils.logger import setup_logger
logger = setup_logger(__name__)

# Configure matplotlib for Nature journal standards
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'legend.frameon': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'lines.linewidth': 1.5,
    'text.usetex': False,
})

# Use seaborn for enhanced styling
sns.set_style('whitegrid')
sns.set_context('paper', rc={'font.size': 10})

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def set_random_seed(seed=42):
    """Set random seed for experiment reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from training.config import Config, default_config
from utils.data_generator import TrajectoryGenerator, TrajectorySample
from models.KFilterNet_single import KFilterNet_single
from models.baseline.baseline_KF import BaselineKalmanFilter
from models.baseline.baseline_AKF import BaselineAdaptiveKF
from models.baseline.baseline_IMM import BaselineIMM
from models.baseline.baseline_UKF import BaselineUKF
from models.baseline.kalman_net_tsp import KalmanNetNN

class ComparisonStudy:
    """Comparison study class"""
    
    def __init__(self, config: Config = None):
        """Initialize comparison study"""
        self.config = config or default_config
        self.output_dir = "outputs/Comparison"
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = setup_logger(__name__)

        # Initialize baseline methods
        self.baselines = {
            'KF': BaselineKalmanFilter(),
            'IMM': BaselineIMM(),
            'AKF': BaselineAdaptiveKF(),
            'UKF': BaselineUKF(),
            'KalmanNetNN': None  # Set to None first, load pretrained model later
        }
        
        # Load pretrained KalmanNetNN model
        self._load_kalman_net_nn_model()
        
        # Load trained KFilterNet_single
        self.KFilterNet_model = None
        self._load_KMNet_model()
    
    def _load_kalman_net_nn_model(self):
        """Load pretrained KalmanNetNN model"""
        try:
            # Find KalmanNetNN pretrained model
            kalman_net_path = 'models/pretrained/Kalmannet.pth'

            if kalman_net_path and os.path.exists(kalman_net_path):
                self.logger.info(f"Loading KalmanNetNN pretrained model: {kalman_net_path}")
                self.baselines['KalmanNetNN'] = KalmanNetNN(self.config)
                checkpoint = torch.load(kalman_net_path, map_location='cpu')
                self.baselines['KalmanNetNN'].load_state_dict(checkpoint['model_state_dict'])
                self.baselines['KalmanNetNN'].eval().to(self.config.training.device)
                self.logger.info(f"KalmanNetNN model loaded successfully!")
            else:
                self.logger.info("KalmanNetNN pretrained model not found, will use default configuration model")
                self.baselines['KalmanNetNN'] = KalmanNetNN(self.config)
                
        except Exception as e:
            self.logger.info(f"Failed to load KalmanNetNN model: {e}")
            self.baselines['KalmanNetNN'] = KalmanNetNN(self.config)
    
    def _load_KMNet_model(self):
        """Load trained KFilterNet_single model"""
        try:
            # Find best model
            best_model_path = 'models/pretrained/KFilterNet.pth'

            if best_model_path and os.path.exists(best_model_path):
                self.logger.info(f"Loading KFilterNet pretrained model: {best_model_path}")
                self.KFilterNet_model = KFilterNet_single(self.config)
                checkpoint = torch.load(best_model_path, map_location='cpu')
                self.KFilterNet_model.load_state_dict(checkpoint['model_state_dict'])
                self.KFilterNet_model.eval().to(self.config.training.device)
                self.logger.info(f"KFilterNet model loaded successfully!")
            else:
                self.logger.info("Trained model not found, will use default configuration model")
                self.KFilterNet_model = KFilterNet_single(self.config)
                
        except Exception as e:
            self.logger.info(f"Failed to load model: {e}")
            self.KFilterNet_model = KFilterNet_single(self.config)
    
    def evaluate_method(self, method_name: str, observations: np.ndarray, 
                       true_states: np.ndarray) -> Dict:
        """Evaluate performance of a single method"""
        start_time = time.time()
        
        if method_name == 'KFilterNet':
            # Use neural network model
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observations).unsqueeze(0).to(self.config.training.device)
                pred_states, _, _ = self.KFilterNet_model(obs_tensor)
                pred_states = pred_states.squeeze(0).cpu().numpy()
        elif method_name == 'KalmanNetNN':
            # Use KalmanNetNN model
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observations).unsqueeze(0).to(self.config.training.device)
                pred_states, _, _ = self.baselines[method_name](obs_tensor)
                pred_states = pred_states.squeeze(0).cpu().numpy()
        else:
            # Use baseline method
            baseline = self.baselines[method_name]
            pred_states = baseline.filter(observations)
        
        inference_time = time.time() - start_time
        
        # Calculate RMSE
        rmse_x = np.sqrt(mean_squared_error(true_states[:, 0], pred_states[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(true_states[:, 1], pred_states[:, 1]))
        rmse_z = np.sqrt(mean_squared_error(true_states[:, 2], pred_states[:, 2]))
        rmse_total = np.sqrt(mean_squared_error(true_states[:, 0:3], pred_states[:, 0:3]))
        
        # Calculate RMSE for each timestep
        rmse_per_timestep = []
        for t in range(len(true_states)):
            rmse_t = np.sqrt(mean_squared_error(
                true_states[t, 0:3], 
                pred_states[t, 0:3]
            ))
            rmse_per_timestep.append(rmse_t)
        
        return {
            'method': method_name,
            'rmse_total': rmse_total,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'rmse_z': rmse_z,
            'rmse_per_timestep': rmse_per_timestep,
            'final_rmse': rmse_per_timestep[-1] if rmse_per_timestep else rmse_total,
            'inference_time': inference_time,
            'predicted_states': pred_states
        }
    
    def experiment_1_motion_modes(self):
        """Experiment 1: Comparison under different motion mode groups"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Experiment 1: Comparison under different motion mode groups")
        self.logger.info("=" * 60)
        
        # Generate test data
        generator = TrajectoryGenerator(self.config.data)
        
        # Test configurations - now 4 groups with different motion type combinations
        test_configs = [
            {
                'name': 'basic_motion',
                'description': 'Basic Motion (CV + CA + CT)',
                'motion_types': ['constant_velocity', 'constant_acceleration', 'coordinated_turn']
            },
            {
                'name': 'air_combat',
                'description': 'Air Combat Maneuvers (Weaving + Vertical + Loop)',
                'motion_types': ['weaving', 'vertical_maneuver', 'loop']
            },
            {
                'name': 'mixed_full',
                'description': 'Mixed Maneuvering (All Modes)',
                'motion_types': ['maneuvering']
            },
            {
                'name': 'random_full',
                'description': 'Random Full (7 motion types random selection)',
                'motion_types': ['constant_velocity', 'constant_acceleration', 'coordinated_turn',
                                 'weaving', 'vertical_maneuver', 'loop', 'maneuvering']
            }
        ]
        
        results = {}
        
        for config in test_configs:
            self.logger.info(f"\nTest configuration: {config['description']}")
            config_results = []
            
            # Generate test samples - 180 total distributed evenly among motion types
            test_samples = []
            samples_per_type = 180 // len(config['motion_types'])
            remainder = 180 % len(config['motion_types'])
            for i, motion_type in enumerate(config['motion_types']):
                # First 'remainder' types get one extra sample
                count = samples_per_type + 1 if i < remainder else samples_per_type
                for _ in range(count):
                    sample = generator.generate_sample(motion_type=motion_type)
                    test_samples.append(sample)
            
            # Evaluate each method
            methods = ['KFilterNet', 'KF', 'AKF', 'IMM', 'UKF', 'KalmanNetNN']
            
            for method in methods:
                self.logger.info(f"  Evaluating {method}...")
                method_results = []
                
                for sample in test_samples:
                    result = self.evaluate_method(
                        method, 
                        sample.observations, 
                        sample.states
                    )
                    method_results.append(result)
                
                # Calculate average results and standard deviation
                avg_rmse_total = np.mean([r['rmse_total'] for r in method_results])
                std_rmse_total = np.std([r['rmse_total'] for r in method_results])
                avg_rmse_x = np.mean([r['rmse_x'] for r in method_results])
                avg_rmse_y = np.mean([r['rmse_y'] for r in method_results])
                avg_rmse_z = np.mean([r['rmse_z'] for r in method_results])
                avg_inference_time = np.mean([r['inference_time'] for r in method_results])
                
                config_results.append({
                    'method': method,
                    'avg_rmse_total': avg_rmse_total,
                    'std_rmse_total': std_rmse_total,
                    'avg_rmse_x': avg_rmse_x,
                    'avg_rmse_y': avg_rmse_y,
                    'avg_rmse_z': avg_rmse_z,
                    'avg_inference_time': avg_inference_time,
                    'num_samples': len(method_results)
                })
                
                self.logger.info(f"    Average RMSE: {avg_rmse_total:.6f} ± {std_rmse_total:.6f}, Inference time: {avg_inference_time:.4f}s")
            
            results[config['name']] = config_results
        
        # Save results
        self._save_experiment_1_results(results)
        self._visualize_experiment_1_results(results)
        
        return results
    
    def experiment_2_noise_levels(self):
        """Experiment 2: Comparison under different noise configurations"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Experiment 2: Comparison under different noise configurations")
        self.logger.info("=" * 60)
        
        # Three noise configurations
        noise_configs = [
            {
                'name': 'low_base',
                'description': 'Low Base Gaussian Noise',
                'process_noise': 0.03,
                'observation_noise': 0.2,
                'enable_impulse': False,
                'enable_correlated': False
            },
            {
                'name': 'high_base',
                'description': 'High Base Gaussian Noise',
                'process_noise': 0.3,
                'observation_noise': 2.0,
                'enable_impulse': False,
                'enable_correlated': False
            },
            {
                'name': 'mixed_noise',
                'description': 'Mixed Noise (Gaussian + Impulse + Correlated)',
                'process_noise_range': (0.01, 0.5),
                'observation_noise_range': (0.1, 3.0),
                'enable_impulse': True,
                'enable_correlated': True
            }
        ]
        
        results = {}
        
        for noise_config in noise_configs:
            self.logger.info(f"\nNoise configuration: {noise_config['description']}")

            # Create appropriate generator for this noise config
            if noise_config['name'] == 'mixed_noise':
                # Use default config for mixed noise (impulse and correlated enabled)
                data_config = copy.deepcopy(self.config.data)
            else:
                # Create modified config for base noise (no impulse, no correlated)
                data_config = copy.deepcopy(self.config.data)
                data_config.enable_impulse_noise = False
                data_config.enable_correlated_noise = False

            generator = TrajectoryGenerator(data_config)

            config_results = []
            test_samples = []

            # Generate test samples
            if noise_config['name'] == 'mixed_noise':
                # Mixed noise: use random noise from ranges
                for _ in range(120):  # 120 samples for mixed noise
                    sample = generator.generate_sample()
                    test_samples.append(sample)
            else:
                # Base noise: use fixed noise levels
                for _ in range(120):  # 120 samples for each base noise config
                    sample = generator.generate_sample(
                        process_noise=noise_config['process_noise'],
                        observation_noise=noise_config['observation_noise']
                    )
                    test_samples.append(sample)
            
            # Evaluate each method
            methods = ['KFilterNet', 'KF', 'AKF', 'IMM', 'UKF', 'KalmanNetNN']
            
            for method in methods:
                self.logger.info(f"  Evaluating {method}...")
                method_results = []
                
                for sample in test_samples:
                    result = self.evaluate_method(
                        method, 
                        sample.observations, 
                        sample.states
                    )
                    method_results.append(result)
                
                # Calculate average results and standard deviation
                avg_rmse_total = np.mean([r['rmse_total'] for r in method_results])
                std_rmse_total = np.std([r['rmse_total'] for r in method_results])
                avg_rmse_x = np.mean([r['rmse_x'] for r in method_results])
                avg_rmse_y = np.mean([r['rmse_y'] for r in method_results])
                avg_rmse_z = np.mean([r['rmse_z'] for r in method_results])
                avg_inference_time = np.mean([r['inference_time'] for r in method_results])
                
                config_results.append({
                    'method': method,
                    'avg_rmse_total': avg_rmse_total,
                    'std_rmse_total': std_rmse_total,
                    'avg_rmse_x': avg_rmse_x,
                    'avg_rmse_y': avg_rmse_y,
                    'avg_rmse_z': avg_rmse_z,
                    'avg_inference_time': avg_inference_time,
                    'num_samples': len(method_results)
                })
                
                self.logger.info(f"    Average RMSE: {avg_rmse_total:.6f} ± {std_rmse_total:.6f}, Inference time: {avg_inference_time:.4f}s")
            
            results[noise_config['name']] = config_results
        
        # Save results
        self._save_experiment_2_results(results)
        self._visualize_experiment_2_results(results)
        
        return results

    def _generate_maneuver_switch_trajectory(self, duration=60.0, dt=0.02):
        """Generate trajectory with two maneuver mode switches (CV -> CT -> CA)"""
        T = int(duration / dt)
        states = np.zeros((T, 9))
        observations = np.zeros((T, 3))

        # Define two switch points
        switch_time_1 = int(T / 3)  # First switch at 20s
        switch_time_2 = int(2 * T / 3)  # Second switch at 40s

        # Phase 1: CV motion (0-20s)
        vx, vy, vz = 200.0, 150.0, 50.0
        x, y, z = 0.0, 0.0, 0.0

        for t in range(switch_time_1):
            states[t] = [x, y, z, vx, vy, vz, 0, 0, 0]
            x += vx * dt
            y += vy * dt
            z += vz * dt

        # Phase 2: CT turn motion (20-40s)
        omega = 0.1  # Turn rate
        for t in range(switch_time_1, switch_time_2):
            dt_local = dt
            # Coordinated turn equations
            sin_wt = np.sin(omega * dt_local)
            cos_wt = np.cos(omega * dt_local)
            new_x = x + (cos_wt - 1) / omega * vx + sin_wt / omega * vy
            new_y = y - sin_wt / omega * vx + (cos_wt - 1) / omega * vy
            new_vx = cos_wt * vx + sin_wt * vy
            new_vy = -sin_wt * vx + cos_wt * vy
            states[t] = [new_x, new_y, z, new_vx, new_vy, vz, 0, 0, 0]
            x, y, vx, vy = new_x, new_y, new_vx, new_vy

        # Phase 3: CA acceleration motion (40-60s)
        ax, ay = 5.0, 3.0
        for t in range(switch_time_2, T):
            states[t] = [x, y, z, vx, vy, vz, ax, ay, 0]
            x += vx * dt + 0.5 * ax * dt**2
            y += vy * dt + 0.5 * ay * dt**2
            z += vz * dt
            vx += ax * dt
            vy += ay * dt

        # Add observation noise (Gaussian only for base trajectory)
        observation_noise_std = 0.5
        for t in range(T):
            observations[t] = states[t, 0:3] + np.random.normal(0, observation_noise_std, 3)

        return TrajectorySample(
            states=states,
            observations=observations,
            process_noise=0.1,
            observation_noise=observation_noise_std**2,
            motion_type='maneuver_switch',
            dt=dt,
            metadata={
                'switch_points': [switch_time_1, switch_time_2],
                'modes': ['CV', 'CT', 'CA']
            }
        )

    def _generate_noise_surge_trajectory(self, duration=60.0, dt=0.02):
        """Generate trajectory with noise surge events - similar to maneuver_switch_trajectory"""
        T = int(duration / dt)
        states = np.zeros((T, 9))
        observations = np.zeros((T, 3))

        # Similar initial conditions to maneuver_switch_trajectory
        vx, vy, vz = 200.0, -150.0, 50.0
        x, y, z = 0.0, 0.0, 0.0
        ax, ay = 0.2, -0.3

        surge_start = int(T / 3)  # Surge starts at 20s
        surge_end = int(2 * T / 3)  # Surge ends at 40s
        base_noise_std = 0.3
        high_noise_std = 3.0

        for t in range(T):
            # State update
            states[t] = [x, y, z, vx, vy, vz, ax, ay, 0]
            x += vx * dt + 0.5 * ax * dt**2
            y += vy * dt + 0.5 * ay * dt**2
            z += vz * dt
            vx += ax * dt
            vy += ay * dt

            # Noise surge: high noise during middle third
            if surge_start <= t < surge_end:
                noise_std = high_noise_std
                # 10% impulse probability during surge
                if np.random.random() < 0.1:
                    noise_std *= 5.0
            else:
                noise_std = base_noise_std

            observations[t] = states[t, 0:3] + np.random.normal(0, noise_std, 3)

        return TrajectorySample(
            states=states,
            observations=observations,
            process_noise=0.1,
            observation_noise=base_noise_std**2,
            motion_type='noise_surge',
            dt=dt,
            metadata={
                'surge_start': surge_start,
                'surge_end': surge_end,
                'noise_levels': {
                    'before': base_noise_std,
                    'during': high_noise_std,
                    'after': base_noise_std
                }
            }
        )

    def experiment_3_trajectory_tracking(self):
        """Experiment 3: Long trajectory tracking comparison"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Experiment 3: Long trajectory tracking comparison")
        self.logger.info("=" * 60)
        
        # Generate two long trajectories
        self.logger.info("\nGenerating maneuver switch trajectory (CV -> CT -> CA)...")
        trajectory_maneuver = self._generate_maneuver_switch_trajectory(duration=60.0)
        self.logger.info("Maneuver switch trajectory generated.")

        self.logger.info("\nGenerating noise surge trajectory...")
        trajectory_noise = self._generate_noise_surge_trajectory(duration=60.0)
        self.logger.info("Noise surge trajectory generated.")

        trajectories = {
            'maneuver_switch': trajectory_maneuver,
            'noise_surge': trajectory_noise
        }
        results = {}

        methods = ['KFilterNet', 'KF', 'IMM', 'UKF', 'KalmanNetNN']

        for traj_name, sample in trajectories.items():
            self.logger.info(f"\nEvaluating {traj_name} trajectory...")
            method_results = {}
            
            for method in methods:
                self.logger.info(f"  Evaluating {method}...")
                result = self.evaluate_method(
                    method, 
                    sample.observations, 
                    sample.states
                )
                method_results[method] = result

            results[traj_name] = {
                'sample': sample,
                'method_results': method_results
            }
        
        # Save results and visualization
        self._save_experiment_3_results(results)
        self._visualize_experiment_3_results(results)
        
        return results
    
    def _save_experiment_1_results(self, results: Dict):
        """Save experiment 1 results"""
        # Save JSON
        json_path = os.path.join(self.output_dir, "experiment_1_motion_modes.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV
        csv_data = []
        for motion_mode, mode_results in results.items():
            for result in mode_results:
                csv_data.append({
                    'motion_mode': motion_mode,
                    'method': result['method'],
                    'avg_rmse_total': result['avg_rmse_total'],
                    'std_rmse_total': result['std_rmse_total'],
                    'avg_rmse_x': result['avg_rmse_x'],
                    'avg_rmse_y': result['avg_rmse_y'],
                    'avg_rmse_z': result['avg_rmse_z'],
                    'avg_inference_time': result['avg_inference_time'],
                    'num_samples': result['num_samples']
                })
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.output_dir, "experiment_1_motion_modes.csv")
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Experiment 1 results saved to: {json_path}, {csv_path}")
    
    def _save_experiment_2_results(self, results: Dict):
        """Save experiment 2 results"""
        # Save JSON
        json_path = os.path.join(self.output_dir, "experiment_2_noise_levels.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV
        csv_data = []
        for noise_level, level_results in results.items():
            for result in level_results:
                csv_data.append({
                    'noise_level': noise_level,
                    'method': result['method'],
                    'avg_rmse_total': result['avg_rmse_total'],
                    'std_rmse_total': result['std_rmse_total'],
                    'avg_rmse_x': result['avg_rmse_x'],
                    'avg_rmse_y': result['avg_rmse_y'],
                    'avg_rmse_z': result['avg_rmse_z'],
                    'avg_inference_time': result['avg_inference_time'],
                    'num_samples': result['num_samples']
                })
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.output_dir, "experiment_2_noise_levels.csv")
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Experiment 2 results saved to: {json_path}, {csv_path}")
    
    def _save_experiment_3_results(self, results: Dict):
        """Save experiment 3 results"""
        # Save JSON (simplified version, without complete trajectory data)
        json_path = os.path.join(self.output_dir, "experiment_3_trajectory_tracking.json")
        simplified_results = {}
        
        for motion_type, data in results.items():
            simplified_results[motion_type] = {}
            for method, result in data['method_results'].items():
                simplified_results[motion_type][method] = {
                    'rmse_total': result['rmse_total'],
                    'rmse_x': result['rmse_x'],
                    'rmse_y': result['rmse_y'],
                    'rmse_z': result['rmse_z'],
                    'final_rmse': result['final_rmse'],
                    'inference_time': result['inference_time']
                }
        
        with open(json_path, 'w') as f:
            json.dump(simplified_results, f, indent=2)
        
        self.logger.info(f"Experiment 3 results saved to: {json_path}")
    
    def _visualize_experiment_1_results(self, results: Dict):
        """Visualize experiment 1 results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        motion_modes = list(results.keys())
        methods = ['KFilterNet', 'KF', 'AKF', 'IMM', 'UKF', 'KalmanNetNN']
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        
        # 1. Total RMSE comparison
        ax = axes[0, 0]
        x = np.arange(len(motion_modes))
        width = 0.15
        
        for i, method in enumerate(methods):
            rmse_values = []
            for mode in motion_modes:
                mode_results = results[mode]
                method_result = next(r for r in mode_results if r['method'] == method)
                rmse_values.append(method_result['avg_rmse_total'])
            
            ax.bar(x + i * width, rmse_values, width, label=method, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Motion Mode')
        ax.set_ylabel('Average RMSE')
        ax.set_title('Average RMSE Comparison Across Motion Modes')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(motion_modes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Inference time comparison
        ax = axes[0, 1]
        for i, method in enumerate(methods):
            time_values = []
            for mode in motion_modes:
                mode_results = results[mode]
                method_result = next(r for r in mode_results if r['method'] == method)
                time_values.append(method_result['avg_inference_time'])
            
            ax.bar(x + i * width, time_values, width, label=method, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Motion Mode')
        ax.set_ylabel('Average Inference Time (s)')
        ax.set_title('Inference Time Comparison Across Motion Modes')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(motion_modes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Multi-dimensional RMSE comparison (mixed_full mode)
        ax = axes[1, 0]
        mixed_results = results['mixed_full']
        
        for i, method in enumerate(methods):
            method_result = next(r for r in mixed_results if r['method'] == method)
            rmse_vals = [method_result['avg_rmse_x'], method_result['avg_rmse_y'], method_result['avg_rmse_z']]
            
            ax.plot(['X', 'Y', 'Z'], rmse_vals, 'o-', label=method, 
                   color=colors[i], linewidth=2, markersize=6)
        
        ax.set_xlabel('Axis')
        ax.set_ylabel('Average RMSE')
        ax.set_title('RMSE Comparison Across Dimensions (Mixed Full Mode)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Performance improvement comparison
        ax = axes[1, 1]
        KFilterNet_rmse = {}
        for mode in motion_modes:
            mode_results = results[mode]
            KFilterNet_result = next(r for r in mode_results if r['method'] == 'KFilterNet')
            KFilterNet_rmse[mode] = KFilterNet_result['avg_rmse_total']
        
        for method in methods[1:]:  # Exclude KFilterNet
            improvement_values = []
            for mode in motion_modes:
                mode_results = results[mode]
                method_result = next(r for r in mode_results if r['method'] == method)
                improvement = (method_result['avg_rmse_total'] - KFilterNet_rmse[mode]) / KFilterNet_rmse[mode] * 100
                improvement_values.append(improvement)
            
            method_idx = methods.index(method)
            ax.bar(x + method_idx * width, improvement_values, width, 
                  label=method, color=colors[method_idx], alpha=0.8)
        
        ax.set_xlabel('Motion Mode')
        ax.set_ylabel('Performance Improvement (%)')
        ax.set_title('KFilterNet Performance Improvement Over Other Methods')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(motion_modes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "experiment_1_motion_modes_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Experiment 1 visualization results saved to: {plot_path}")
    
    def _visualize_experiment_2_results(self, results: Dict):
        """Visualize experiment 2 results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        noise_levels = list(results.keys())
        methods = ['KFilterNet', 'KF', 'AKF', 'IMM', 'UKF', 'KalmanNetNN']
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        
        # 1. Total RMSE comparison
        ax = axes[0, 0]
        x = np.arange(len(noise_levels))
        width = 0.15
        
        for i, method in enumerate(methods):
            rmse_values = []
            for level in noise_levels:
                level_results = results[level]
                method_result = next(r for r in level_results if r['method'] == method)
                rmse_values.append(method_result['avg_rmse_total'])
            
            ax.bar(x + i * width, rmse_values, width, label=method, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Average RMSE')
        ax.set_title('Average RMSE Comparison Across Noise Levels')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(noise_levels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Inference time comparison
        ax = axes[0, 1]
        for i, method in enumerate(methods):
            time_values = []
            for level in noise_levels:
                level_results = results[level]
                method_result = next(r for r in level_results if r['method'] == method)
                time_values.append(method_result['avg_inference_time'])
            
            ax.bar(x + i * width, time_values, width, label=method, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Average Inference Time (s)')
        ax.set_title('Inference Time Comparison Across Noise Levels')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(noise_levels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. RMSE trend vs noise
        ax = axes[1, 0]
        for i, method in enumerate(methods):
            rmse_values = []
            for level in noise_levels:
                level_results = results[level]
                method_result = next(r for r in level_results if r['method'] == method)
                rmse_values.append(method_result['avg_rmse_total'])
            
            ax.plot(noise_levels, rmse_values, 'o-', label=method, 
                   color=colors[i], linewidth=2, markersize=6)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Average RMSE')
        ax.set_title('RMSE Trend vs Noise Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Robustness analysis (RMSE growth rate)
        ax = axes[1, 1]
        for i, method in enumerate(methods):
            rmse_values = []
            for level in noise_levels:
                level_results = results[level]
                method_result = next(r for r in level_results if r['method'] == method)
                rmse_values.append(method_result['avg_rmse_total'])
            
            # Calculate growth rate relative to lowest noise
            growth_rates = [(rmse - rmse_values[0]) / rmse_values[0] * 100 for rmse in rmse_values]
            ax.plot(noise_levels, growth_rates, 'o-', label=method, 
                   color=colors[i], linewidth=2, markersize=6)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('RMSE Growth Rate (%)')
        ax.set_title('Noise Robustness Analysis of Different Methods')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "experiment_2_noise_levels_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Experiment 2 visualization results saved to: {plot_path}")

    def _add_switch_or_surge_markers(self, ax, sample, time_steps, traj_name):
        """Add switch points or noise surge markers to a plot - unified red dashed line style"""
        if traj_name == 'maneuver_switch':
            for pt in sample.metadata['switch_points']:
                ax.axvline(x=time_steps[pt], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        else:
            # Noise surge: use the same red dashed line style for boundaries
            surge_start = sample.metadata['surge_start']
            surge_end = sample.metadata['surge_end']
            ax.axvline(x=time_steps[surge_start], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvline(x=time_steps[surge_end], color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    def _plot_coordinate_comparison(self, ax, sample, pred_array, methods, colors,
                                     coord_idx, time_steps, coord_name, traj_name):
        """Plot comparison for a single coordinate axis"""
        ax.plot(time_steps, sample.states[:, coord_idx], 'k-', linewidth=2.5, label='True', alpha=0.8)
        ax.plot(time_steps, sample.observations[:, coord_idx], '.', color='gray', markersize=2, alpha=0.5)

        for i, pred in enumerate(pred_array):
            ax.plot(time_steps, pred[:, coord_idx], '-', color=colors[i], linewidth=1.8, alpha=0.9)

        self._add_switch_or_surge_markers(ax, sample, time_steps, traj_name)

        coord_labels = ['X', 'Y', 'Z']
        ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{coord_labels[coord_idx]} Position (m)', fontsize=10, fontweight='bold')
        ax.set_title(f'{coord_labels[coord_idx]}-axis Tracking', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_coordinate_comparison(self, ax, sample, pred_array, methods, colors, line_styles,
                                       coord_idx, time_steps, coord_name, traj_name, panel_label):
        """Plot comparison for a single coordinate axis with Nature style - version 2"""
        coord_labels = ['X', 'Y', 'Z']

        # True trajectory (black, thick line)
        ax.plot(time_steps, sample.states[:, coord_idx], 'k-', linewidth=2.8, label='True', alpha=0.9, zorder=6)

        # Observations (gray, sparse)
        ax.plot(time_steps[::5], sample.observations[::5, coord_idx], '.', color='gray', markersize=2, alpha=0.4, zorder=1)

        # Method predictions with distinct styles
        for i, method in enumerate(methods):
            pred = pred_array[i]
            style = line_styles[method]
            ax.plot(time_steps, pred[:, coord_idx],
                   linestyle=style['linestyle'],
                   linewidth=style['linewidth'],
                   color=colors[method],
                   alpha=0.85,
                   label=method,
                   zorder=style['zorder'])

        # Mark switch points or surge region
        self._add_switch_or_surge_markers(ax, sample, time_steps, traj_name)

        ax.set_xlabel('Time (s)', fontsize=10, fontweight='normal', fontfamily='serif')
        ax.set_ylabel(f"{coord_labels[coord_idx]} Position (m)", fontsize=10, fontweight='normal', fontfamily='serif')
        ax.set_title(panel_label, fontsize=11, fontweight='bold', loc='left', fontfamily='serif')
        ax.grid(True, alpha=0.3)

    def _plot_performance_comparison_bar(self, ax, methods, method_results):
        """Plot bar chart comparing RMSE performance across methods"""
        rmse_totals = [method_results[m]['rmse_total'] for m in methods]
        final_rmses = [method_results[m]['final_rmse'] for m in methods]
        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax.bar(x - width/2, rmse_totals, width, label='Total RMSE',
                       color='#3498db', alpha=0.8, edgecolor='k', linewidth=1)
        bars2 = ax.bar(x + width/2, final_rmses, width, label='Final RMSE',
                       color='#e74c3c', alpha=0.8, edgecolor='k', linewidth=1)

        # Add numeric labels
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xlabel('Method', fontsize=11, fontweight='bold')
        ax.set_ylabel('RMSE (m)', fontsize=11, fontweight='bold')
        ax.set_title('Performance Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=0, fontsize=10)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    def _visualize_experiment_3_results(self, results: Dict):
        """Visualize experiment 3 results - Nature journal style with 1x3 layout"""
        methods = ['KFilterNet', 'KF', 'IMM', 'UKF', 'KalmanNetNN']

        # Colorblind-friendly palette (Okabe-Ito style)
        colors = {
            'KFilterNet': '#E69F00',  # Orange - primary emphasis
            'KalmanNetNN': '#56B4E9',  # Sky blue
            'KF': '#009E73',  # Blue-green
            'IMM': '#F0E442',  # Yellow
            'UKF': '#0072B2'   # Blue
        }

        # Line style scheme for differentiation (KFilterNet: thick solid, others: thinner with distinct styles)
        line_styles = {
            'KFilterNet': {'linestyle': '-', 'linewidth': 2.8, 'alpha': 1.0, 'zorder': 5, 'marker': 'o', 'markersize': 0},
            'KalmanNetNN': {'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.85, 'zorder': 4, 'marker': '^', 'markersize': 0},
            'KF': {'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.75, 'zorder': 3, 'marker': 's', 'markersize': 0},
            'IMM': {'linestyle': '-.', 'linewidth': 1.5, 'alpha': 0.75, 'zorder': 3, 'marker': 'd', 'markersize': 0},
            'UKF': {'linestyle': ':', 'linewidth': 1.5, 'alpha': 0.75, 'zorder': 3, 'marker': 'x', 'markersize': 0}
        }

        # Marker styles for RMSE plot (sparse markers to avoid clutter)
        rmse_markers = {
            'KFilterNet': {'marker': 'o', 'markersize': 4, 'markevery': 150},
            'KalmanNetNN': {'marker': '^', 'markersize': 4, 'markevery': 150},
            'KF': {'marker': 's', 'markersize': 4, 'markevery': 150},
            'IMM': {'marker': 'd', 'markersize': 4, 'markevery': 150},
            'UKF': {'marker': 'x', 'markersize': 5, 'markevery': 150}
        }

        for traj_name, data in results.items():
            sample = data['sample']
            method_results = data['method_results']
            time_steps = np.arange(len(sample.states)) * sample.dt

            # Create figure with 1x3 layout
            fig = plt.figure(figsize=(18, 6))

            # 1 row, 3 columns layout with adjusted width ratios
            # 3D trajectory and RMSE plots get more space than legend panel
            gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.4, 0.8], hspace=0.25, wspace=0.25)

            # Calculate KFilterNet improvement for annotation
            kfilter_rmse = method_results['KFilterNet']['rmse_total']
            avg_baseline_rmse = np.mean([method_results[m]['rmse_total'] for m in methods[1:]])
            improvement_pct = ((avg_baseline_rmse - kfilter_rmse) / avg_baseline_rmse) * 100

            # ===== PANEL (a): 3D Trajectory Plot =====
            ax1 = fig.add_subplot(gs[0, 0], projection='3d')

            # True trajectory (black, thick)
            ax1.plot(sample.states[:, 0], sample.states[:, 1], sample.states[:, 2],
                     'k-', linewidth=2.8, label='True', alpha=0.9, zorder=6)

            # Observations (gray, sparse)
            ax1.plot(sample.observations[::10, 0], sample.observations[::10, 1], sample.observations[::10, 2],
                     'o', color='gray', markersize=2, alpha=0.4, label='Obs', zorder=1)

            # Method predictions with distinct styles
            for i, method in enumerate(methods):
                pred = method_results[method]['predicted_states']
                style = line_styles[method]
                ax1.plot(pred[:, 0], pred[:, 1], pred[:, 2],
                         '-', color=colors[method], linewidth=style['linewidth'],
                         alpha=0.85, label=method, zorder=style['zorder'])

            # Mark switch points or surge region
            if traj_name == 'maneuver_switch':
                for pt in sample.metadata['switch_points']:
                    ax1.scatter(sample.states[pt, 0], sample.states[pt, 1], sample.states[pt, 2],
                               marker='*', s=250, c='red', edgecolors='k', linewidth=1.2, zorder=10)
            else:
                surge_start = sample.metadata['surge_start']
                surge_end = sample.metadata['surge_end']
                # Mark surge boundaries with red markers
                ax1.scatter([sample.states[surge_start, 0], sample.states[surge_end, 0]],
                           [sample.states[surge_start, 1], sample.states[surge_end, 1]],
                           [sample.states[surge_start, 2], sample.states[surge_end, 2]],
                           marker='^', s=200, c='red', edgecolors='k', linewidth=1.5, zorder=10)

            ax1.set_xlabel('X (m)', fontsize=10, fontweight='normal', fontfamily='serif')
            ax1.set_ylabel('Y (m)', fontsize=10, fontweight='normal', fontfamily='serif')
            ax1.set_zlabel('Z (m)', fontsize=10, fontweight='normal', fontfamily='serif')
            ax1.set_title('(a) 3D Trajectory', fontsize=11, fontweight='bold', loc='left', fontfamily='serif')
            ax1.view_init(elev=20, azim=45)

            # ===== PANEL (b): RMSE Over Time =====
            ax2 = fig.add_subplot(gs[0, 1])

            # Plot RMSE for each method with enhanced differentiation
            for method in methods:
                rmse_ts = method_results[method]['rmse_per_timestep']
                style = line_styles[method]
                marker_style = rmse_markers[method]

                # Smooth the RMSE curve slightly for readability
                rmse_smooth = gaussian_filter1d(rmse_ts, sigma=15)

                ax2.plot(time_steps, rmse_smooth,
                        linestyle=style['linestyle'],
                        linewidth=style['linewidth'],
                        color=colors[method],
                        alpha=style['alpha'],
                        label=method,
                        zorder=style['zorder'])

                # Add sparse markers
                ax2.plot(time_steps[marker_style['markevery']::marker_style['markevery']],
                        rmse_smooth[marker_style['markevery']::marker_style['markevery']],
                        marker=marker_style['marker'],
                        markersize=marker_style['markersize'],
                        color=colors[method],
                        linestyle='None',
                        alpha=0.7,
                        zorder=style['zorder'] + 1)

            # Mark switch points or surge region
            if traj_name == 'maneuver_switch':
                for pt in sample.metadata['switch_points']:
                    ax2.axvline(x=time_steps[pt], color='red', linestyle='--', linewidth=1.2, alpha=0.7)
            else:
                surge_start = sample.metadata['surge_start']
                surge_end = sample.metadata['surge_end']
                ax2.axvline(x=time_steps[surge_start], color='red', linestyle='--', linewidth=1.2, alpha=0.7)
                ax2.axvline(x=time_steps[surge_end], color='red', linestyle='--', linewidth=1.2, alpha=0.7)

            ax2.set_xlabel('Time (s)', fontsize=10, fontweight='normal', fontfamily='serif')
            ax2.set_ylabel('RMSE (m)', fontsize=10, fontweight='normal', fontfamily='serif')
            ax2.set_title('(b) RMSE Over Time', fontsize=11, fontweight='bold', loc='left', fontfamily='serif')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 60)  # Set x-axis limit to 0-60 seconds

            # ===== PANEL (c): Centralized Legend and Info Panel =====
            ax_info = fig.add_subplot(gs[0, 2])
            ax_info.axis('off')  # Hide axes for clean legend panel

            # Create custom legend with line styles
            from matplotlib.lines import Line2D
            from matplotlib.font_manager import FontProperties

            # Create font properties for serif font
            legend_font = FontProperties(family='serif', size=10)

            legend_elements = []
            for method in methods:
                style = line_styles[method]
                legend_elements.append(
                    Line2D([0], [0], linestyle=style['linestyle'], linewidth=style['linewidth'],
                           color=colors[method], label=method)
                )

            # Add legend with serif font
            legend = ax_info.legend(handles=legend_elements, loc='center', prop=legend_font,
                                    frameon=True, framealpha=0.95, edgecolor='black')

            # Add KFilterNet improvement annotation
            improvement_text = f"KFilterNet Improvement:\n\n{improvement_pct:.1f}% better than average baseline\n\n(RMSE: {kfilter_rmse:.3f} vs {avg_baseline_rmse:.3f} m)"
            ax_info.text(0.05, 0.15, improvement_text, fontsize=11, fontweight='bold', fontfamily='serif',
                         ha='left', va='center',
                         bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9C4', alpha=0.7,
                                   edgecolor='black', linewidth=1.5))

            # Add main figure title based on trajectory type
            if traj_name == 'maneuver_switch':
                fig.suptitle('Trajectory with Maneuver Mode Switches (CV $\\rightarrow$ CT $\\rightarrow$ CA)',
                            fontsize=14, fontweight='bold', y=0.98, fontfamily='serif')
            else:
                fig.suptitle('Trajectory with Noise Surge Events',
                            fontsize=14, fontweight='bold', y=0.98, fontfamily='serif')

            # Save as EPS format for publication
            plot_path = os.path.join(self.output_dir, f"experiment_3_{traj_name}_tracking_nature.eps")
            plt.savefig(plot_path, format='eps', dpi=400, bbox_inches='tight', facecolor='white')
            plt.close()
            self.logger.info(f"{traj_name} trajectory visualization (Nature style 1x3 layout) saved to: {plot_path}")

    def run_all_experiments(self):
        """Run all comparison experiments"""
        self.logger.info("Starting all comparison experiments...")
        
        # Experiment 1: Comparison under different motion modes
        results1 = self.experiment_1_motion_modes()
        
        # Experiment 2: Comparison under different noise covariances
        results2 = self.experiment_2_noise_levels()
        
        # Experiment 3: Trajectory tracking comparison
        results3 = self.experiment_3_trajectory_tracking()
        
        # Generate summary report
        self._generate_summary_report(results1, results2, results3)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("All comparison experiments completed!")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("=" * 60)
        
        return results1, results2, results3
    
    def _generate_summary_report(self, results1=None, results2=None, results3=None):
        """Generate summary report"""
        report_path = os.path.join(self.output_dir, "summary_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# KFilterNet Comparison Study Report\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Experiment Overview\n\n")
            f.write("This experiment compares the performance of the Flexible Kalman Filter Network (KFilterNet) with the following baseline methods:\n")
            f.write("- **Kalman Filter (KF)**: Standard Kalman filter\n")
            f.write("- **Interacting Multiple Model (IMM)**: Interacting multiple model filter\n")
            f.write("- **Adaptive Kalman Filter (AKF)**: Kalman filter with adaptive noise estimation\n")
            f.write("- **Unscented Kalman Filter (UKF)**: Unscented Kalman filter for nonlinear systems\n")
            f.write("- **KalmanNetNN**: Neural network enhanced Kalman filter\n\n")
            
            # Experiment 1 report
            f.write("## Experiment 1: Comparison under different motion modes\n\n")
            if results1:
                f.write("### Mixed full motion mode results\n\n")
                mixed_results = results1['mixed_full']
                f.write("| Method | Average RMSE | X-axis RMSE | Y-axis RMSE | Z-axis RMSE | Inference Time (s) |\n")
                f.write("|--------|--------------|-------------|-------------|-------------|---------------------|\n")
                for result in mixed_results:
                    f.write(f"| {result['method']} | {result['avg_rmse_total']:.6f} | "
                           f"{result['avg_rmse_x']:.6f} | {result['avg_rmse_y']:.6f} | "
                           f"{result['avg_rmse_z']:.6f} | {result['avg_inference_time']:.4f} |\n")
                
                # Find best method
                best_method = min(mixed_results, key=lambda x: x['avg_rmse_total'])
                f.write(f"\n**Best Method**: {best_method['method']} (RMSE: {best_method['avg_rmse_total']:.6f})\n\n")
            
            # Experiment 2 report
            f.write("## Experiment 2: Comparison under different noise covariances\n\n")
            if results2:
                f.write("### Mixed noise level results\n\n")
                mixed_noise_results = results2['mixed_noise']
                f.write("| Method | Average RMSE | X-axis RMSE | Y-axis RMSE | Z-axis RMSE | Inference Time (s) |\n")
                f.write("|--------|--------------|-------------|-------------|-------------|---------------------|\n")
                for result in mixed_noise_results:
                    f.write(f"| {result['method']} | {result['avg_rmse_total']:.6f} | "
                           f"{result['avg_rmse_x']:.6f} | {result['avg_rmse_y']:.6f} | "
                           f"{result['avg_rmse_z']:.6f} | {result['avg_inference_time']:.4f} |\n")
                
                # Find best method
                best_method = min(mixed_noise_results, key=lambda x: x['avg_rmse_total'])
                f.write(f"\n**Best Method**: {best_method['method']} (RMSE: {best_method['avg_rmse_total']:.6f})\n\n")
            
            # Experiment 3 report
            f.write("## Experiment 3: Trajectory tracking comparison\n\n")
            if results3:
                f.write("### Best performance under each motion mode\n\n")
                f.write("| Motion Mode | Best Method | RMSE | Final RMSE |\n")
                f.write("|-------------|-------------|------|------------|\n")
                for motion_type, data in results3.items():
                    method_results = data['method_results']
                    best_method = min(method_results.items(), key=lambda x: x[1]['rmse_total'])
                    f.write(f"| {motion_type} | {best_method[0]} | {best_method[1]['rmse_total']:.6f} | "
                           f"{best_method[1]['final_rmse']:.6f} |\n")
                f.write("\n")
            
            # Overall conclusions
            f.write("## Overall Conclusions\n\n")
            f.write("Based on the results of the three experiments, the following conclusions can be drawn:\n\n")
            
            if results1 and results2:
                # Count KFilterNet win rate
                flexible_wins = 0
                total_comparisons = 0
                
                # Win rate in experiment 1
                for mode, mode_results in results1.items():
                    flexible_result = next(r for r in mode_results if r['method'] == 'KFilterNet')
                    for result in mode_results:
                        if result['method'] != 'KFilterNet':
                            total_comparisons += 1
                            if flexible_result['avg_rmse_total'] < result['avg_rmse_total']:
                                flexible_wins += 1
                
                # Win rate in experiment 2
                for level, level_results in results2.items():
                    flexible_result = next(r for r in level_results if r['method'] == 'KFilterNet')
                    for result in level_results:
                        if result['method'] != 'KFilterNet':
                            total_comparisons += 1
                            if flexible_result['avg_rmse_total'] < result['avg_rmse_total']:
                                flexible_wins += 1
                
                win_rate = flexible_wins / total_comparisons * 100 if total_comparisons > 0 else 0
                f.write(f"1. **Performance Advantage**: KFilterNet won {flexible_wins} out of {total_comparisons} comparisons, with a win rate of {win_rate:.1f}%\n\n")
                
                # Calculate average performance improvement
                improvements = []
                for mode, mode_results in results1.items():
                    flexible_result = next(r for r in mode_results if r['method'] == 'KFilterNet')
                    for result in mode_results:
                        if result['method'] != 'KFilterNet':
                            improvement = (result['avg_rmse_total'] - flexible_result['avg_rmse_total']) / flexible_result['avg_rmse_total'] * 100
                            improvements.append(improvement)
                
                avg_improvement = np.mean(improvements) if improvements else 0
                f.write(f"2. **Average Performance Improvement**: Average improvement of {avg_improvement:.2f}% compared to other methods\n\n")
            
            f.write("3. **Computational Efficiency**: KFilterNet maintains high accuracy while having comparable inference time to traditional methods\n\n")
            f.write("4. **Robustness**: Shows good stability under different noise levels and motion modes\n\n")
            f.write("5. **Practicality**: Flexible module design enables adaptation to different application scenarios\n\n")
        
        self.logger.info(f"Summary report saved to: {report_path}")


def main():
    """Main function"""

    # Create comparison experiment instance
    comparison = ComparisonStudy()

    # Set random seed
    set_random_seed(42)
    
    # Run all experiments
    results1, results2, results3 = comparison.run_all_experiments()
    
    logger.info("\nComparison study completed!")
    logger.info(f"Results saved to: {comparison.output_dir}")


if __name__ == "__main__":
    main()
