import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
import os

# Add project root to path for constants import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from constants import (
    STABILITY_EPSILON,
    LOG_SOFTPLUS_MIN,
    LOG_SOFTPLUS_MAX,
    CORRELATION_COEFF_MIN,
    CORRELATION_COEFF_MAX,
    COVARIANCE_CLAMP_MIN,
    COVARIANCE_CLAMP_MAX,
)

class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class NoiseNetwork(nn.Module):
    """Noise prediction network"""
    def __init__(self, 
                 input_dim: int,
                 state_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super(NoiseNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Network predicts log of noise (ensures positivity)
        self.network = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim,  # Predict diagonal elements
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict diagonal elements of noise covariance matrix"""
        log_noise = self.network(x)  # [B, state_dim]
        
        # Limit log_noise range to prevent exp explosion
        
        # Use softplus instead of exp for better numerical stability
        noise = torch.nn.functional.softplus(log_noise)
        
        # Add minimum value constraint to prevent covariance from being too small
        
        return noise

class StateTransitionNetwork(nn.Module):
    """State transition matrix prediction network"""
    def __init__(self,
                 input_dim: int,
                 state_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super(StateTransitionNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Predict residual ΔF
        self.network = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim * state_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict state transition matrix F = base_F + ΔF"""
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Predict residual
        delta_F_flat = self.network(x)  # [B, state_dim * state_dim]
        F = delta_F_flat.view(batch_size, seq_len, self.state_dim, self.state_dim)

        # Stability constraint: ensure eigenvalues are within unit circle
        # F = self._enforce_stability(F)
        
        return F
    
    def _enforce_stability(self, F: torch.Tensor, epsilon: float = STABILITY_EPSILON) -> torch.Tensor:
        """Enforce stability constraint"""
        # Scale matrix to ensure spectral radius < 1
        batch_size, seq_len = F.shape[0], F.shape[1]
        
        # Check if input contains inf or NaN
        if torch.isinf(F).any() or torch.isnan(F).any():
            print("Warning: State transition matrix F contains inf or NaN, using identity matrix instead")
            # Return identity matrix as safe fallback
            eye = torch.eye(self.state_dim, device=F.device, dtype=F.dtype)
            return eye.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        
        # Create output tensor to avoid in-place operations
        F_stable = F.clone()
        
        for i in range(batch_size):
            for j in range(seq_len):
                try:
                    # Compute eigenvalues
                    eigenvalues = torch.linalg.eigvals(F[i, j])
                    
                    # Check if eigenvalues contain inf or NaN
                    if torch.isinf(eigenvalues).any() or torch.isnan(eigenvalues).any():
                        print(f"Warning: Eigenvalue computation result contains inf or NaN (batch={i}, seq={j})")
                        # Use identity matrix for this matrix
                        F_stable[i, j] = torch.eye(self.state_dim, device=F.device, dtype=F.dtype)
                        continue
                    
                    max_eigenvalue = torch.max(torch.abs(eigenvalues))
                    
                    if max_eigenvalue > 1 - epsilon:
                        scaling_factor = (1 - epsilon) / (max_eigenvalue + epsilon)
                        F_stable[i, j] = F[i, j] * scaling_factor
                except RuntimeError as e:
                    print(f"Warning: Eigenvalue computation failed (batch={i}, seq={j}): {e}")
                    # Use identity matrix for this matrix
                    F_stable[i, j] = torch.eye(self.state_dim, device=F.device, dtype=F.dtype)
        
        return F_stable

class KalmanGainNetwork(nn.Module):
    """Kalman gain prediction network"""
    def __init__(self,
                 input_dim: int,
                 state_dim: int,
                 obs_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super(KalmanGainNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # Directly predict Kalman gain
        self.network = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim * obs_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict Kalman gain matrix"""
        batch_size, seq_len = x.shape[0], x.shape[1]
        K_flat = self.network(x)  # [B, state_dim * obs_dim]
        K = K_flat.view(batch_size, seq_len, self.state_dim, self.obs_dim)
        return K

class ObservationMatrixNetwork(nn.Module):
    """Observation matrix prediction network"""
    def __init__(self,
                 input_dim: int,
                 state_dim: int,
                 obs_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super(ObservationMatrixNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # Predict observation matrix H
        self.network = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=obs_dim * state_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict observation matrix H = base_H + ΔH"""
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        delta_H_flat = self.network(x)  # [B, obs_dim * state_dim]
        H = delta_H_flat.view(batch_size, seq_len, self.obs_dim, self.state_dim)

        return H

class HistoryEncoder(nn.Module):
    """History information encoder"""
    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super(HistoryEncoder, self).__init__()
        
        # Use GRU to encode history information
        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                observations: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode history information
        
        Args:
            observations: [B, T, obs_dim]
            lengths: [B] Actual sequence length
        
        Returns:
            encoded: [B, hidden_dim]
        """

        _, hidden = self.gru(observations)  # hidden: [num_layers, B, hidden_dim]
        
        # Use hidden state of last layer
        encoded = hidden[-1]  # [B, hidden_dim]
        encoded = self.dropout(encoded)
        
        return encoded

class CovarianceNetwork(nn.Module):
    """Covariance matrix prediction network"""
    def __init__(self,
                 input_dim: int,
                 state_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super(CovarianceNetwork, self).__init__()

        self.state_dim = state_dim

        # Predict diagonal elements and correlation coefficients of covariance matrix
        self.network = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim + state_dim * (state_dim - 1) // 2,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict covariance matrix"""
        batch_size = x.shape[0]

        # Network output
        output = self.network(x)  # [B, state_dim + state_dim*(state_dim-1)/2]

        # Parse diagonal elements and correlation coefficients
        diag = output[:, :self.state_dim]  # [B, state_dim]
        corr_coeffs = output[:, self.state_dim:]  # [B, state_dim*(state_dim-1)/2]

        # Construct covariance matrix
        cov_matrix = torch.zeros(batch_size, self.state_dim, self.state_dim, device=x.device)

        # Limit diag range to prevent exp explosion
        diag = torch.clamp(diag, min=LOG_SOFTPLUS_MIN, max=LOG_SOFTPLUS_MAX)

        # Diagonal elements (variance) - use softplus instead of exp
        cov_matrix[:, torch.arange(self.state_dim), torch.arange(self.state_dim)] = torch.nn.functional.softplus(diag)

        # Off-diagonal elements (covariance)
        if self.state_dim > 1:
            triu_indices = torch.triu_indices(self.state_dim, self.state_dim, offset=1)
            # Limit correlation coefficient range
            corr_coeffs = torch.clamp(corr_coeffs, min=CORRELATION_COEFF_MIN, max=CORRELATION_COEFF_MAX)
            cov_matrix[:, triu_indices[0], triu_indices[1]] = corr_coeffs
            cov_matrix[:, triu_indices[1], triu_indices[0]] = corr_coeffs
        
        # Ensure positive definiteness
        cov_matrix = self._ensure_positive_definite(cov_matrix)
        
        return cov_matrix
    
    def _ensure_positive_definite(self, cov_matrix: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
        """Ensure covariance matrix is positive definite"""
        # Add small diagonal matrix
        eye = torch.eye(self.state_dim, device=cov_matrix.device).unsqueeze(0)
        cov_matrix = cov_matrix + epsilon * eye
        
        # Limit covariance matrix value range
        cov_matrix = torch.clamp(cov_matrix, min=COVARIANCE_CLAMP_MIN, max=COVARIANCE_CLAMP_MAX)
        
        return cov_matrix


class InitialStateComputer(nn.Module):
    """Initial state calculator (using first 3 observations)"""
    def __init__(self, state_dim: int, obs_dim: int, dt: float = 0.1):
        super(InitialStateComputer, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt

    def forward(self, initial_obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate initial state and covariance using first 3 observations
        
        Args:
            initial_obs_seq: [B, 3, obs_dim] First 3 observations (may be normalized)
            
        Returns:
            state: [B, state_dim] Initial state
            cov: [B, state_dim, state_dim] Initial covariance
        """
        batch_size = initial_obs_seq.shape[0]

        # Extract position observations
        pos_0 = initial_obs_seq[:, 0, :3]  # Position at t=0
        pos_1 = initial_obs_seq[:, 1, :3]  # Position at t=1
        pos_2 = initial_obs_seq[:, 2, :3]  # Position at t=2
        
        # Calculate velocity (using central difference)
        velocity = (pos_2 - pos_0) / (2 * self.dt)  # [B, 3]
        
        # Calculate acceleration
        acc = (pos_2 - 2 * pos_1 + pos_0) / (self.dt ** 2)  # [B, 3]
        
        # Construct initial state [position, velocity, acceleration, other states]
        state = torch.zeros(batch_size, self.state_dim, device=initial_obs_seq.device)
        state[:, 0:3] = pos_1  # Use second observation as initial position
        state[:, 3:6] = velocity  # Velocity
        state[:, 6:9] = acc  # Acceleration
        
        # Calculate initial covariance (based on observation noise)
        # Assume position observation noise standard deviation is 1.0, velocity and acceleration noise are larger
        cov_diag = torch.ones(batch_size, self.state_dim, device=initial_obs_seq.device)
        cov_diag[:, 0:3] = 1.0  # Position noise
        cov_diag[:, 3:6] = 5.0  # Velocity noise (larger estimation error)
        cov_diag[:, 6:9] = 10.0  # Acceleration noise (even larger estimation error)
        
        # Diagonal covariance matrix
        cov = torch.diag_embed(cov_diag)  # [B, state_dim, state_dim]

        return state, cov
