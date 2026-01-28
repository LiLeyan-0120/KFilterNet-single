import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Dict, Optional
import os
from mpl_toolkits.mplot3d import Axes3D

def visualize_trajectory_comparison(results: List[Dict], save_path: str = None):
    """
    Visualize trajectory comparison
    
    Args:
        results: List containing prediction results and true values
        save_path: Save path
    """
    num_samples = len(results)
    fig = plt.figure(figsize=(20, 6 * num_samples))
    
    for idx, result in enumerate(results):
        observations = result['observations']
        states_pred = result['states_pred']
        states_true = result['states_true']
        
        # Create subplots
        ax1 = fig.add_subplot(num_samples, 3, idx * 3 + 1, projection='3d')
        ax2 = fig.add_subplot(num_samples, 3, idx * 3 + 2)
        ax3 = fig.add_subplot(num_samples, 3, idx * 3 + 3)
        
        # 3D trajectory
        ax1.plot(states_true[:, 0], states_true[:, 1], states_true[:, 2], 
                'g-', label='True', linewidth=2, alpha=0.8)
        ax1.plot(states_pred[:, 0], states_pred[:, 1], states_pred[:, 2], 
                'b--', label='Predicted', linewidth=2, alpha=0.8)
        ax1.plot(observations[:, 0], observations[:, 1], observations[:, 2], 
                'r.', label='Observations', alpha=0.6, markersize=4)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'Sample {idx + 1}: 3D Trajectory')
        ax1.legend()
        ax1.grid(True)
        
        # Position error
        time = np.arange(len(states_true))
        pos_error = np.linalg.norm(states_pred[:, :3] - states_true[:, :3], axis=1)
        
        ax2.plot(time, pos_error, 'b-', linewidth=2)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title(f'Sample {idx + 1}: Position Error')
        ax2.grid(True)
        
        # Velocity error
        vel_error = np.linalg.norm(states_pred[:, 3:6] - states_true[:, 3:6], axis=1)
        
        ax3.plot(time, vel_error, 'r-', linewidth=2)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Velocity Error (m/s)')
        ax3.set_title(f'Sample {idx + 1}: Velocity Error')
        ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory comparison saved to {save_path}")
    
    return fig

def visualize_training_results(train_history: Dict, val_history: Dict, save_dir: str):
    """
    Visualize training results
    
    Args:
        train_history: Training history
        val_history: Validation history
        save_dir: Save directory
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss curves
    loss_names = ['loss', 'state_loss', 'covariance_loss', 'kalman_constraint_loss']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, loss_name in enumerate(loss_names):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        if loss_name in train_history and train_history[loss_name]:
            ax.plot(train_history[loss_name], label='Train', alpha=0.8)
        
        if loss_name in val_history and val_history[loss_name]:
            ax.plot(val_history[loss_name], label='Val', alpha=0.8)
        
        ax.set_title(loss_name.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot learning rate curve
    if 'learning_rate' in train_history:
        plt.figure(figsize=(8, 6))
        plt.plot(train_history['learning_rate'], 'b-', linewidth=2)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()

def visualize_kalman_gains(kalman_gains: np.ndarray, save_path: str = None):
    """
    Visualize Kalman gains
    
    Args:
        kalman_gains: [T, state_dim, obs_dim] Kalman gain sequence
        save_path: Save path
    """
    T, state_dim, obs_dim = kalman_gains.shape
    
    fig, axes = plt.subplots(state_dim, obs_dim, figsize=(4 * obs_dim, 3 * state_dim))
    
    if state_dim == 1 and obs_dim == 1:
        axes = np.array([[axes]])
    elif state_dim == 1:
        axes = axes.reshape(1, -1)
    elif obs_dim == 1:
        axes = axes.reshape(-1, 1)
    
    time = np.arange(T)
    
    for i in range(state_dim):
        for j in range(obs_dim):
            ax = axes[i, j]
            ax.plot(time, kalman_gains[:, i, j], 'b-', linewidth=2)
            ax.set_title(f'K[{i},{j}]')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Gain Value')
            ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Kalman gains visualization saved to {save_path}")
    
    return fig

def visualize_covariance_matrices(covariances: np.ndarray, save_path: str = None, max_plots: int = 5):
    """
    Visualize covariance matrices
    
    Args:
        covariances: [T, state_dim, state_dim] Covariance matrix sequence
        save_path: Save path
        max_plots: Maximum number of plots
    """
    T, state_dim, _ = covariances.shape
    
    # Select time points to plot
    plot_indices = np.linspace(0, T - 1, min(max_plots, T), dtype=int)
    
    fig, axes = plt.subplots(1, len(plot_indices), figsize=(4 * len(plot_indices), 4))
    
    if len(plot_indices) == 1:
        axes = [axes]
    
    for idx, t in enumerate(plot_indices):
        ax = axes[idx]
        im = ax.imshow(covariances[t], cmap='hot', interpolation='nearest')
        ax.set_title(f'Covariance at t={t}')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Covariance matrices visualization saved to {save_path}")
    
    return fig

def visualize_noise_levels(process_noise: np.ndarray, observation_noise: np.ndarray, save_path: str = None):
    """
    Visualize noise levels
    
    Args:
        process_noise: [T, state_dim] Process noise
        observation_noise: [T, obs_dim] Observation noise
        save_path: Save path
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    time = np.arange(len(process_noise))
    
    # Process noise
    for i in range(process_noise.shape[1]):
        ax1.plot(time, process_noise[:, i], label=f'Dim {i}', alpha=0.8)
    
    ax1.set_title('Process Noise Levels')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Noise Level')
    ax1.legend()
    ax1.grid(True)
    
    # Observation noise
    for i in range(observation_noise.shape[1]):
        ax2.plot(time, observation_noise[:, i], label=f'Dim {i}', alpha=0.8)
    
    ax2.set_title('Observation Noise Levels')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Noise Level')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Noise levels visualization saved to {save_path}")
    
    return fig

def visualize_state_components(states_true: np.ndarray, states_pred: np.ndarray, save_path: str = None):
    """
    Visualize state component comparison
    
    Args:
        states_true: [T, state_dim] True states
        states_pred: [T, state_dim] Predicted states
        save_path: Save path
    """
    T, state_dim = states_true.shape
    
    # State component names
    state_names = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz', 'Ax', 'Ay', 'Az']
    state_names = state_names[:state_dim]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    time = np.arange(T)
    
    for i in range(min(state_dim, 9)):
        ax = axes[i]
        ax.plot(time, states_true[:, i], 'g-', label='True', linewidth=2, alpha=0.8)
        ax.plot(time, states_pred[:, i], 'b--', label='Predicted', linewidth=2, alpha=0.8)
        ax.set_title(f'State: {state_names[i]}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
    
    # Hide unused subplots
    for i in range(state_dim, 9):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"State components visualization saved to {save_path}")
    
    return fig

def create_summary_plot(results: List[Dict], train_history: Dict, val_history: Dict, save_dir: str):
    """
    Create comprehensive summary plot
    
    Args:
        results: Test results list
        train_history: Training history
        val_history: Validation history
        save_dir: Save directory
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Training curves
    visualize_training_results(train_history, val_history, save_dir)
    
    # 2. Trajectory comparison
    if results:
        visualize_trajectory_comparison(
            results, 
            save_path=os.path.join(save_dir, 'trajectory_comparison.png')
        )
        
        # 3. Kalman gains
        if 'info' in results[0] and 'kalman_gains' in results[0]['info']:
            kalman_gains = results[0]['info']['kalman_gains'][0]  # First sample
            visualize_kalman_gains(
                kalman_gains,
                save_path=os.path.join(save_dir, 'kalman_gains.png')
            )
        
        # 4. Covariance matrices
        if 'covariances' in results[0]:
            covariances = results[0]['covariances']  # First sample
            visualize_covariance_matrices(
                covariances,
                save_path=os.path.join(save_dir, 'covariance_matrices.png')
            )
        
        # 5. State components
        visualize_state_components(
            results[0]['states_true'],
            results[0]['states_pred'],
            save_path=os.path.join(save_dir, 'state_components.png')
        )

def plot_error_distribution(errors: np.ndarray, save_path: str = None):
    """
    Plot error distribution
    
    Args:
        errors: [N] Error array
        save_path: Save path
    """
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Cumulative distribution
    plt.subplot(1, 2, 2)
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cumulative, 'b-', linewidth=2)
    plt.title('Cumulative Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution saved to {save_path}")
    
    return plt.gcf()
