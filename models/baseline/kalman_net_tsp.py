"""
KalmanNet implementation - based on KalmanNet_TSP-main project
Reproduces GRU-based Kalman gain learning architecture
Adapted to current project interface
"""

import torch
import torch.nn as nn
import torch.nn.functional as func
from typing import Optional, Tuple, Dict

class KalmanNetNN(torch.nn.Module):
    """
    KalmanNet neural network - GRU-based Kalman gain learning
    Reproduced from KalmanNet_TSP-main project
    Adapted to current project's Trainer interface
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Get parameters from config
        self.state_dim = config.model.state_dim
        self.obs_dim = config.model.obs_dim
        self.seq_len = config.data.sequence_length
        self.batch_size = config.training.batch_size
        
        # Device
        self.device = config.training.device
        
        # Create system model
        self._create_system_model()
        
        # Build network
        self._build_network()
        
    def _create_system_model(self):
        """Create system model"""
        # Create default system matrices
        # State transition matrix F
        F = torch.eye(self.state_dim)
        dt = 0.1  # Time step
        
        # Position <- velocity
        if self.state_dim >= 6:
            F[0:3, 3:6] = torch.eye(3) * dt
        
        # Position <- acceleration
        if self.state_dim >= 9:
            F[0:3, 6:9] = 0.5 * torch.eye(3) * dt ** 2
            F[3:6, 6:9] = torch.eye(3) * dt
        
        # Observation matrix H
        H = torch.zeros(self.obs_dim, self.state_dim)
        H[0:self.obs_dim, 0:self.obs_dim] = torch.eye(self.obs_dim)
        
        # Noise covariances
        Q = torch.eye(self.state_dim) * 0.1  # Process noise
        R = torch.eye(self.obs_dim) * 0.5    # Observation noise
        
        # Prior covariances
        prior_Q = torch.eye(self.state_dim)
        prior_Sigma = torch.zeros(self.state_dim, self.state_dim)
        prior_S = torch.eye(self.obs_dim)
        
        # Store as buffers
        self.register_buffer('F', F)
        self.register_buffer('H', H)
        self.register_buffer('Q', Q)
        self.register_buffer('R', R)
        self.register_buffer('prior_Q', prior_Q)
        self.register_buffer('prior_Sigma', prior_Sigma)
        self.register_buffer('prior_S', prior_S)
        
        # Initial state
        self.register_buffer('m1x_0', torch.zeros(self.state_dim, 1))
        
    def _build_network(self):
        """Build network structure"""
        # Network parameters
        self.in_mult_KNet = getattr(self.config.model, 'in_mult_KNet', 5)
        self.out_mult_KNet = getattr(self.config.model, 'out_mult_KNet', 40)
        
        self.seq_len_input = 1  # KNet calculates time-step by time-step

        # GRU to track Q
        self.d_input_Q = self.state_dim * self.in_mult_KNet
        self.d_hidden_Q = self.state_dim ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.state_dim * self.in_mult_KNet
        self.d_hidden_Sigma = self.state_dim ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)
       
        # GRU to track S
        self.d_input_S = self.obs_dim ** 2 + 2 * self.obs_dim * self.in_mult_KNet
        self.d_hidden_S = self.obs_dim ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)
        
        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.obs_dim ** 2
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.ReLU()).to(self.device)

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.obs_dim * self.state_dim
        self.d_hidden_FC2 = self.d_input_FC2 * self.out_mult_KNet
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2)).to(self.device)

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.state_dim ** 2
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.ReLU()).to(self.device)

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU()).to(self.device)
        
        # Fully connected 5
        self.d_input_FC5 = self.state_dim
        self.d_output_FC5 = self.state_dim * self.in_mult_KNet
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.ReLU()).to(self.device)

        # Fully connected 6
        self.d_input_FC6 = self.state_dim
        self.d_output_FC6 = self.state_dim * self.in_mult_KNet
        self.FC6 = nn.Sequential(
                nn.Linear(self.d_input_FC6, self.d_output_FC6),
                nn.ReLU()).to(self.device)
        
        # Fully connected 7
        self.d_input_FC7 = 2 * self.obs_dim
        self.d_output_FC7 = 2 * self.obs_dim * self.in_mult_KNet
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.ReLU()).to(self.device)
                
    def f(self, x):
        """State transition function"""
        batched_F = self.F.to(x.device).view(1, self.F.shape[0], self.F.shape[1]).expand(x.shape[0], -1, -1)
        return torch.bmm(batched_F, x)
    
    def h(self, x):
        """Observation function"""
        batched_H = self.H.to(x.device).view(1, self.H.shape[0], self.H.shape[1]).expand(x.shape[0], -1, -1)
        return torch.bmm(batched_H, x)
        
    def InitSequence(self, M1_0, T):
        """
        Initialize sequence
        
        Args:
            M1_0: 1st moment of x at time 0 [batch_size, m, 1]
            T: Sequence length
        """
        self.T = T
        self.m1x_posterior = M1_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)
        
    def step_prior(self):
        """Compute prior"""
        self.m1x_prior = self.f(self.m1x_posterior)
        self.m1y = self.h(self.m1x_prior)
        
    def step_kalman_gain_est(self, y):
        """Kalman gain estimation"""
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_previous, 2) 
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)
        fw_evol_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_posterior_previous, 2)
        fw_update_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_prior_previous, 2)

        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)

        kalman_gain = self._kalman_gain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        current_batch_size = getattr(self, 'current_batch_size', self.batch_size)
        self.kalman_gain = torch.reshape(kalman_gain, (current_batch_size, self.state_dim, self.obs_dim))
        
    def KNet_step(self, y):
        """Kalman Net single step"""
        self.step_prior()
        self.step_kalman_gain_est(y)
        dy = y - self.m1y
        innovation = torch.bmm(self.kalman_gain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + innovation
        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y
        return self.m1x_posterior
        
    def _kalman_gain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        """Kalman gain calculation step"""
        def expand_dim(x):
            current_batch_size = getattr(self, 'current_batch_size', self.batch_size)
            expanded = torch.empty(self.seq_len_input, current_batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        # FC 5
        in_FC5 = fw_update_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        # FC 6
        in_FC6 = fw_evol_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2
        
    def forward(self, observations: torch.Tensor, initial_obs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass - adapted to current project interface
        
        Args:
            observations: [B, T, obs_dim] Observation sequence
            initial_obs: [B, obs_dim] Initial observation (optional)
        
        Returns:
            states: [B, T, state_dim] Estimated state sequence
            covariances: [B, T, state_dim, state_dim] Covariance matrix sequence (returns identity matrix here)
            info: Dict Intermediate information
        """
        batch_size, seq_len, obs_dim = observations.shape
        
        # Set current batch size
        self.current_batch_size = batch_size
        
        # Initialize storage
        states = torch.zeros(batch_size, seq_len, self.state_dim, device=self.device)
        
        # Initialize sequence
        if initial_obs is not None:
            # Use initial observation to estimate initial state
            initial_state = torch.zeros(batch_size, self.state_dim, 1, device=self.device)
            initial_state[:, 0:obs_dim, 0] = initial_obs
        else:
            initial_state = self.m1x_0.reshape(1, self.state_dim, 1).expand(batch_size, -1, -1)
        
        self.InitSequence(initial_state, seq_len)
        self.init_hidden_KNet()
        
        # Adjust batch_size
        self.batch_size = batch_size
        
        # Forward pass
        for t in range(seq_len):
            y_t = observations[:, t, :].reshape(batch_size, obs_dim, 1)
            state_t = self.KNet_step(y_t)
            states[:, t, :] = state_t.squeeze(2)
        
        # Return identity covariance (since KalmanNet_TSP doesn't directly predict covariance)
        covariances = torch.eye(self.state_dim, device=self.device).unsqueeze(0).unsqueeze(0)
        covariances = covariances.expand(batch_size, seq_len, -1, -1)
        
        # Collect intermediate information
        info = {
            'kalman_gains': self.kalman_gain if hasattr(self, 'kalman_gain') else None,
            'innovations': None,
            'observation_matrices': self.H.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1),
            'transition_matrices': self.F.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        }
        
        return states, covariances, info
        
    def init_hidden_KNet(self):
        """Initialize hidden states"""
        weight = next(self.parameters()).data
        # Use current actual batch_size instead of fixed value at initialization
        current_batch_size = getattr(self, 'current_batch_size', self.batch_size)
        hidden = weight.new(self.seq_len_input, current_batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, current_batch_size, 1)
        hidden = weight.new(self.seq_len_input, current_batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, current_batch_size, 1)
        hidden = weight.new(self.seq_len_input, current_batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, current_batch_size, 1)


class SystemModel:
    """
    System model - reproduced from KalmanNet_TSP-main
    Stores system dynamics parameters and generates data
    """
    
    def __init__(self, F, Q, H, R, T, T_test, prior_Q=None, prior_Sigma=None, prior_S=None):
        ####################
        ### Motion Model ###
        ####################
        self.F = F
        self.m = self.F.size()[0]
        self.Q = Q

        #########################
        ### Observation Model ###
        #########################
        self.H = H
        self.n = self.H.size()[0]
        self.R = R

        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S
        

    def f(self, x):
        """State transition function"""
        batched_F = self.F.to(x.device).view(1, self.F.shape[0], self.F.shape[1]).expand(x.shape[0], -1, -1)
        return torch.bmm(batched_F, x)
    
    def h(self, x):
        """Observation function"""
        batched_H = self.H.to(x.device).view(1, self.H.shape[0], self.H.shape[1]).expand(x.shape[0], -1, -1)
        return torch.bmm(batched_H, x)
        
    def InitSequence(self, m1x_0, m2x_0):
        """Initialize sequence"""
        self.m1x_0 = m1x_0
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        """Initialize batched sequence"""
        self.m1x_0_batch = m1x_0_batch
        self.x_prev = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch

    def UpdateCovariance_Matrix(self, Q, R):
        """Update covariance matrices"""
        self.Q = Q
        self.R = R
