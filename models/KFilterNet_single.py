import torch
import torch.nn as nn
import sys
import os

# Add project root to path for constants import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import ModuleType, KALMAN_GAIN_REGULARIZATION
from utils.logger import setup_logger

logger = setup_logger(__name__)

from .components.kalman_components import (
    NoiseNetwork, StateTransitionNetwork, KalmanGainNetwork,
    ObservationMatrixNetwork, HistoryEncoder, CovarianceNetwork, MLP,
    InitialStateComputer
)

class KFilterNet_single(nn.Module):
    """
    Highly configurable Kalman filter network
    Supports flexible selection of each module for convenient ablation experiments
    """
    
    def __init__(self, config):
        super(KFilterNet_single, self).__init__()
        
        self.config = config
        self.state_dim = config.model.state_dim
        self.obs_dim = config.model.obs_dim
        self.hidden_dim = config.model.hidden_dim
        
        # Read type of each module from configuration
        self.H_type = getattr(config.model, 'H_type', ModuleType.LEARNABLE.value)
        self.K_type = getattr(config.model, 'K_type', ModuleType.LEARNABLE.value)
        self.F_type = getattr(config.model, 'F_type', ModuleType.LEARNABLE.value)
        self.Q_type = getattr(config.model, 'Q_type', ModuleType.LEARNABLE.value)
        self.R_type = getattr(config.model, 'R_type', ModuleType.LEARNABLE.value)
        self.init_type = getattr(config.model, 'init_type', ModuleType.LEARNABLE.value)
        
        # History encoder (shared)
        self.history_encoder = HistoryEncoder(
            state_dim=self.state_dim,
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            dropout=config.model.dropout
        )
        
        # 1. Observation matrix H
        self._init_H_module()
        
        # 2. Kalman gain K
        self._init_K_module()
        
        # 3. State transition matrix F
        self._init_F_module()
        
        # 4. Process noise Q
        self._init_Q_module()
        
        # 5. Observation noise R
        self._init_R_module()
        
        # 6. Initial state
        self._init_initial_module()

        # Print configuration information
        self._print_config()
    
    def _init_H_module(self):
        """Initialize observation matrix H module"""
        if self.H_type == ModuleType.LEARNABLE.value:
            # Full network method: learnable H
            self.H_net = ObservationMatrixNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout
            )
            self.H_base = None
            
        elif self.H_type == ModuleType.FIXED.value:
            # Fixed method: only observe position
            self.register_buffer('H_base', self._create_fixed_observation_matrix())
            self.H_net = None
            
        elif self.H_type == ModuleType.SEMI_FIXED.value:
            # Semi-fixed method: base + residual
            self.register_buffer('H_base', self._create_fixed_observation_matrix())
            self.H_net = nn.Linear(self.hidden_dim, self.obs_dim * self.state_dim)
            
        else:  # HYBRID
            # Hybrid method: network prediction + mathematical constraints
            self.H_net = ObservationMatrixNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout
            )
            self.H_base = None
    
    def _init_K_module(self):
        """Initialize Kalman gain K module"""
        if self.K_type == ModuleType.LEARNABLE.value:
            # Full network method: directly learn K
            self.K_net = KalmanGainNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout
            )
            self.use_learned_K = True
            
        elif self.K_type == ModuleType.FIXED.value:
            # Mathematical calculation method
            self.K_net = None
            self.use_learned_K = False
            
        else:  # SEMI_FIXED or HYBRID
            # Network prediction based, mathematical calculation as constraint
            self.K_net = KalmanGainNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,  # Shallower network
                dropout=self.config.model.dropout
            )
            self.use_learned_K = True
    
    def _init_F_module(self):
        """Initialize state transition matrix F module"""
        if self.F_type == ModuleType.LEARNABLE.value:
            # Full network method
            self.F_net = StateTransitionNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout
            )
            self.F_base = None
            
        elif self.F_type == ModuleType.FIXED.value:
            # Fixed method
            self.register_buffer('F_base', self._create_fixed_transition_matrix())
            self.F_net = None
            
        else:  # SEMI_FIXED or HYBRID
            # Semi-fixed method
            self.register_buffer('F_base', self._create_fixed_transition_matrix())
            self.F_net = StateTransitionNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                dropout=self.config.model.dropout
            )
    
    def _init_Q_module(self):
        """Initialize process noise Q module"""
        if self.Q_type == ModuleType.LEARNABLE.value:
            self.Q_net = NoiseNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout
            )
            self.Q_fixed = None
            
        elif self.Q_type == ModuleType.FIXED.value:
            self.Q_net = None
            self.register_buffer('Q_fixed', torch.eye(self.state_dim) * 0.1)
            
        else:  # SEMI_FIXED - Partially learnable mode
            # Fixed base noise + learnable residual
            self.register_buffer('Q_fixed', torch.eye(self.state_dim) * 0.1)  # Base noise
            self.Q_net = NoiseNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                hidden_dim=self.hidden_dim,
                num_layers=3,  # Shallower network, learn residual
                dropout=self.config.model.dropout
            )
    
    def _init_R_module(self):
        """Initialize observation noise R module"""
        if self.R_type == ModuleType.LEARNABLE.value:
            self.R_net = NoiseNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout
            )
            self.R_fixed = None
            
        elif self.R_type == ModuleType.FIXED.value:
            self.R_net = None
            self.register_buffer('R_fixed', torch.eye(self.obs_dim) * 0.5)
            
        else:  # SEMI_FIXED - Partially learnable mode
            # Fixed base noise + learnable residual
            self.register_buffer('R_fixed', torch.eye(self.obs_dim) * 0.5)  # Base noise
            self.R_net = NoiseNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                num_layers=3,  # Shallower network, learn residual
                dropout=self.config.model.dropout
            )
    
    def _init_initial_module(self):
        """Initialize initial state module"""
        if self.init_type == ModuleType.LEARNABLE.value:
            # Network predicts initial state
            self.initial_state_net = MLP(
                input_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.state_dim,
                num_layers=2,
                dropout=self.config.model.dropout
            )
            self.initial_cov_net = CovarianceNetwork(
                input_dim=self.obs_dim,
                state_dim=self.state_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                dropout=self.config.model.dropout
            )
            self.initial_state_computer = None
            
        else:  # FIXED or SEMI_FIXED
            # Use observations to compute initial state
            self.initial_state_net = None
            self.initial_cov_net = None
            # Initial state computation module (use first 3 observations)
            self.initial_state_computer = InitialStateComputer(
                state_dim=self.state_dim,
                obs_dim=self.obs_dim,
                dt=0.1  # Time step
            )
    
    def _print_config(self):
        """Print configuration information"""
        logger.info("\n" + "="*50)
        logger.info("KFilterNet_single Configuration:")
        logger.info("="*50)
        logger.info(f"  State dimension: {self.state_dim}")
        logger.info(f"  Observation dimension: {self.obs_dim}")
        logger.info(f"  Hidden dimension: {self.hidden_dim}")
        logger.info(f"  K module type: {self.K_type}")
        logger.info(f"  H module type: {self.H_type}")
        logger.info(f"  F module type: {self.F_type}")
        logger.info(f"  Q module type: {self.Q_type}")
        logger.info(f"  R module type: {self.R_type}")
        logger.info(f"  Initial state type: {self.init_type}")
        logger.info("="*50 + "\n")
    
    def _create_fixed_observation_matrix(self):
        """Create fixed observation matrix"""
        H = torch.zeros(self.obs_dim, self.state_dim)
        H[0:3, 0:3] = torch.eye(3)
        return H
    
    def _create_fixed_transition_matrix(self):
        """Create fixed state transition matrix"""
        F = torch.eye(self.state_dim)
        dt = self.config.data.dt
        F[0:3, 3:6] = torch.eye(3) * dt
        F[0:3, 6:9] = 0.5 * torch.eye(3) * dt ** 2
        F[3:6, 6:9] = torch.eye(3) * dt
        return F
    
    def forward(self, observations, initial_obs=None):
        """Forward propagation"""
        batch_size, seq_len, _ = observations.shape
        
        # Encode historical information
        hist_encoded_all = self._encode_history(observations)
        
        # Predict each matrix
        H_all = self._predict_H(hist_encoded_all, batch_size, seq_len)
        F_all = self._predict_F(hist_encoded_all, batch_size, seq_len)
        Q_all = self._predict_Q(hist_encoded_all, batch_size, seq_len)
        R_all = self._predict_R(hist_encoded_all, batch_size, seq_len)
        
        # Initial state
        state_pred, cov_pred = self._compute_initial_state(observations)
        
        # Kalman gain
        if self.use_learned_K:
            K_all = self._predict_K(hist_encoded_all, batch_size, seq_len)
        else:
            K_all = None
        
        # Kalman filter
        states, covariances, K_actual = self._kalman_filter(
            observations, state_pred, cov_pred,
            H_all, F_all, Q_all, R_all, K_all
        )
        
        # Collect intermediate information
        info = {
            'kalman_gains': K_actual,
            'observation_matrices': H_all,
            'transition_matrices': F_all,
            'process_noise': Q_all,
            'observation_noise': R_all
        }
        
        return states, covariances, info
    
    def _encode_history(self, observations):
        """Encode historical information"""
        batch_size, seq_len, _ = observations.shape
        
        # Simplify: directly use GRU to encode observation sequence
        h0 = torch.zeros(self.history_encoder.gru.num_layers, batch_size,
                        self.hidden_dim, device=observations.device)
        output, _ = self.history_encoder.gru(observations, h0)
        
        return output
    
    def _predict_H(self, hist_encoded, batch_size, seq_len):
        """Predict observation matrix H"""
        if self.H_type == ModuleType.LEARNABLE.value:
            return self.H_net(hist_encoded)
        elif self.H_type == ModuleType.FIXED.value:
            return self.H_base.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        else:  # SEMI_FIXED
            H_delta = self.H_net(hist_encoded).view(batch_size, seq_len, self.obs_dim, self.state_dim)
            H_base = self.H_base.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
            return H_base + H_delta
    
    def _predict_K(self, hist_encoded, batch_size, seq_len):
        """Predict Kalman gain K"""
        if self.K_type in [ModuleType.LEARNABLE.value, ModuleType.HYBRID.value]:
            return self.K_net(hist_encoded)
        else:
            return None
    
    def _predict_F(self, hist_encoded, batch_size, seq_len):
        """Predict state transition matrix F"""
        if self.F_type == ModuleType.LEARNABLE.value:
            return self.F_net(hist_encoded)
        elif self.F_type == ModuleType.FIXED.value:
            return self.F_base.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        else:  # SEMI_FIXED
            F_delta = self.F_net(hist_encoded)
            F_base = self.F_base.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
            return F_base + F_delta
    
    def _predict_Q(self, hist_encoded, batch_size, seq_len):
        """Predict process noise Q"""
        if self.Q_type == ModuleType.LEARNABLE.value:
            Q_diag = self.Q_net(hist_encoded)
            return torch.diag_embed(Q_diag)
        elif self.Q_type == ModuleType.FIXED.value:
            return self.Q_fixed.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        else:  # SEMI_FIXED - Partially learnable mode
            # Base noise + residual
            Q_base = self.Q_fixed.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
            Q_residual_diag = self.Q_net(hist_encoded)  # Network predicts residual
            Q_residual = torch.diag_embed(Q_residual_diag)
            return Q_base + Q_residual
    
    def _predict_R(self, hist_encoded, batch_size, seq_len):
        """Predict observation noise R"""
        if self.R_type == ModuleType.LEARNABLE.value:
            R_diag = self.R_net(hist_encoded)
            return torch.diag_embed(R_diag)
        elif self.R_type == ModuleType.FIXED.value:
            return self.R_fixed.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        else:  # SEMI_FIXED - Partially learnable mode
            # Base noise + residual
            R_base = self.R_fixed.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
            R_residual_diag = self.R_net(hist_encoded)  # Network predicts residual
            R_residual = torch.diag_embed(R_residual_diag)
            return R_base + R_residual
    
    def _compute_initial_state(self, observations):
        """Compute initial state"""
        if self.init_type == ModuleType.LEARNABLE.value:
            # Network prediction
            initial_obs = observations[:, 0]
            state_pred = self.initial_state_net(initial_obs)
            cov_pred = self.initial_cov_net(initial_obs)
            return state_pred, cov_pred
        else:
            # Observation computation
            if observations.shape[1] >= 3:
                initial_obs_seq = observations[:, :3, :]
                return self.initial_state_computer(initial_obs_seq)
            else:
                # Sequence too short, use simple initialization
                initial_obs = observations[:, 0]
                state_pred = torch.zeros(initial_obs.shape[0], self.state_dim, device=observations.device)
                cov_pred = torch.eye(self.state_dim, device=observations.device).unsqueeze(0).expand(initial_obs.shape[0], -1, -1) * 0.1
                return state_pred, cov_pred
    
    def _kalman_filter(self, observations, state_pred, cov_pred, H_all, F_all, Q_all, R_all, K_all):
        """Kalman filter"""
        batch_size, seq_len, _ = observations.shape
        
        states = torch.zeros(batch_size, seq_len, self.state_dim, device=observations.device)
        covariances = torch.zeros(batch_size, seq_len, self.state_dim, self.state_dim, device=observations.device)
        kalman_gains = torch.zeros(batch_size, seq_len, self.state_dim, self.obs_dim, device=observations.device)
        
        for t in range(seq_len):
            observation_t = observations[:, t]
            H_t = H_all[:, t]
            F_t = F_all[:, t]
            Q_t = Q_all[:, t]
            R_t = R_all[:, t]
            
            # Calculate Kalman gain
            if K_all is not None:
                K_t = K_all[:, t]  # Network predicted K
            else:
                K_t = self._compute_kalman_gain(cov_pred, H_t, R_t)  # Mathematically calculated K
            
            # Update step
            innovation = observation_t.unsqueeze(2) - torch.bmm(H_t, state_pred.unsqueeze(2))
            state_updated = state_pred + torch.bmm(K_t, innovation).squeeze(2)
            
            # Covariance update
            I_KH = torch.eye(self.state_dim, device=observations.device).unsqueeze(0) - torch.bmm(K_t, H_t)
            cov_updated = torch.bmm(torch.bmm(I_KH, cov_pred), I_KH.transpose(1, 2)) + \
                         torch.bmm(torch.bmm(K_t, R_t), K_t.transpose(1, 2))
            
            # Store results
            states[:, t] = state_updated
            covariances[:, t] = cov_updated
            kalman_gains[:, t] = K_t
            
            # Prediction step
            state_pred = torch.bmm(F_t, state_updated.unsqueeze(2)).squeeze(2)
            cov_pred = torch.bmm(torch.bmm(F_t, cov_updated), F_t.transpose(1, 2)) + Q_t
        
        return states, covariances, kalman_gains
    
    def _compute_kalman_gain(self, cov_pred, H_t, R_t):
        """Mathematically calculate Kalman gain (numerically stable version)"""
        HPHR = torch.bmm(torch.bmm(H_t, cov_pred), H_t.transpose(1, 2)) + R_t
        
        # Add regularization to improve numerical stability
        HPHR_reg = HPHR + KALMAN_GAIN_REGULARIZATION * torch.eye(self.obs_dim, device=H_t.device).unsqueeze(0)
        
        # Use solve instead of inv (more stable)
        try:
            K_t = torch.linalg.solve(
                HPHR_reg.transpose(-2, -1),
                torch.bmm(cov_pred, H_t.transpose(1, 2)).transpose(-2, -1)
            ).transpose(-2, -1)
        except:
            # Backup plan: use pseudo-inverse
            K_t = torch.bmm(torch.bmm(cov_pred, H_t.transpose(1, 2)), torch.linalg.pinv(HPHR_reg))
        
        return K_t
    
    def predict_step(self, state, cov, observation, history_states, history_obs):
        """Single step prediction (for online prediction)"""
        batch_size = state.shape[0]
        
        # Encode historical information
        lengths = torch.ones(batch_size, device=state.device) * history_states.shape[1]
        hist_encoded = self.history_encoder(history_states, history_obs, lengths)
        
        # Predict parameters
        H_t = self._predict_H(hist_encoded, batch_size, 1).squeeze(1)
        F_t = self._predict_F(hist_encoded, batch_size, 1).squeeze(1)
        Q_t = self._predict_Q(hist_encoded, batch_size, 1).squeeze(1)
        R_t = self._predict_R(hist_encoded, batch_size, 1).squeeze(1)
        
        if self.use_learned_K:
            K_t = self._predict_K(hist_encoded, batch_size, 1).squeeze(1)
        else:
            K_t = self._compute_kalman_gain(cov, H_t, R_t)
        
        # Update step
        innovation = observation.unsqueeze(2) - torch.bmm(H_t, state.unsqueeze(2))
        state_updated = state + torch.bmm(K_t, innovation).squeeze(2)
        
        # Covariance update
        I_KH = torch.eye(self.state_dim, device=state.device).unsqueeze(0) - torch.bmm(K_t, H_t)
        cov_updated = torch.bmm(torch.bmm(I_KH, cov), I_KH.transpose(1, 2)) + \
                     torch.bmm(torch.bmm(K_t, R_t), K_t.transpose(1, 2))
        
        return state_updated, cov_updated, K_t
