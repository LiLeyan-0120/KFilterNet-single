import torch
import torch.nn as nn

from constants import ModuleType
from .components.kalman_components import (
    NoiseNetwork, StateTransitionNetwork, KalmanGainNetwork,
    ObservationMatrixNetwork, CovarianceNetwork, MLP,
    InitialStateComputer
)

class KFilterNet_single_stepInput(nn.Module):
    """
    KalmanNet implementation according to paper scheme 1
    Uses paper-defined input features (F1-F4) instead of raw observation history
    Maintains module flexibility of KFilterNet model
    """

    def __init__(self, config):
        super(KFilterNet_single_stepInput, self).__init__()

        self.config = config
        self.state_dim = config.model.state_dim
        self.obs_dim = config.model.obs_dim
        self.hidden_dim = config.model.hidden_dim

        # Read module types from configuration
        self.H_type = getattr(config.model, 'H_type', ModuleType.LEARNABLE.value)
        self.K_type = getattr(config.model, 'K_type', ModuleType.LEARNABLE.value)
        self.F_type = getattr(config.model, 'F_type', ModuleType.LEARNABLE.value)
        self.Q_type = getattr(config.model, 'Q_type', ModuleType.LEARNABLE.value)
        self.R_type = getattr(config.model, 'R_type', ModuleType.LEARNABLE.value)
        self.init_type = getattr(config.model, 'init_type', ModuleType.LEARNABLE.value)

        # Feature processor
        # Input feature dimensions: F1 + F2 + F4 (or F1 + F3 + F4)
        # F1: obs_dim, F2: obs_dim, F3: state_dim, F4: state_dim
        feature_dim = self.obs_dim + self.obs_dim + self.state_dim  # Use F1, F2, F4 combination

        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout)
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
            self.H_net = ObservationMatrixNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout
            )
            self.H_base = None
            print(f"  H module: Full network method (learnable)")

        elif self.H_type == ModuleType.FIXED.value:
            self.register_buffer('H_base', self._create_fixed_observation_matrix())
            self.H_net = None
            print(f"  H module: Fixed matrix (position observation only)")

        elif self.H_type == ModuleType.SEMI_FIXED.value:
            self.register_buffer('H_base', self._create_fixed_observation_matrix())
            self.H_net = nn.Linear(self.hidden_dim, self.obs_dim * self.state_dim)
            print(f"  H module: Semi-fixed (base + residual learning)")

        else:  # HYBRID
            self.H_net = ObservationMatrixNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout
            )
            self.H_base = None
            print(f"  H module: Hybrid method (network + constraints)")

    def _init_K_module(self):
        """Initialize Kalman gain K module"""
        if self.K_type == ModuleType.LEARNABLE.value:
            self.K_net = KalmanGainNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout
            )
            self.use_learned_K = True
            print(f"  K module: Full network method (direct learning)")

        elif self.K_type == ModuleType.FIXED.value:
            self.K_net = None
            self.use_learned_K = False
            print(f"  K module: Mathematical computation (no network)")

        else:  # SEMI_FIXED or HYBRID
            self.K_net = KalmanGainNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                dropout=self.config.model.dropout
            )
            self.use_learned_K = True
            print(f"  K module: Hybrid method (network prediction + mathematical constraints)")

    def _init_F_module(self):
        """Initialize state transition matrix F module"""
        if self.F_type == ModuleType.LEARNABLE.value:
            self.F_net = StateTransitionNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout
            )
            self.F_base = None
            print(f"  F module: Full network method (learnable)")

        elif self.F_type == ModuleType.FIXED.value:
            self.register_buffer('F_base', self._create_fixed_transition_matrix())
            self.F_net = None
            print(f"  F module: Fixed matrix (constant velocity model)")

        else:  # SEMI_FIXED or HYBRID
            self.register_buffer('F_base', self._create_fixed_transition_matrix())
            self.F_net = StateTransitionNetwork(
                input_dim=self.hidden_dim,
                state_dim=self.state_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                dropout=self.config.model.dropout
            )
            print(f"  F module: Semi-fixed (base + residual learning)")

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
            print(f"  Q module: Full network method (learnable)")

        else:  # FIXED
            self.Q_net = None
            self.register_buffer('Q_fixed', torch.eye(self.state_dim) * 0.1)
            print(f"  Q module: Fixed matrix")

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
            print(f"  R module: Full network method (learnable)")

        else:  # FIXED
            self.R_net = None
            self.register_buffer('R_fixed', torch.eye(self.obs_dim) * 0.5)
            print(f"  R module: Fixed matrix")

    def _init_initial_module(self):
        """Initialize initial state module"""
        if self.init_type == ModuleType.LEARNABLE.value:
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
            print(f"  Initial state: Network prediction")

        else:  # FIXED or SEMI_FIXED
            self.initial_state_net = None
            self.initial_cov_net = None
            self.initial_state_computer = InitialStateComputer(
                state_dim=self.state_dim,
                obs_dim=self.obs_dim,
                dt=0.1
            )
            print(f"  Initial state: Observation computation")

    def _print_config(self):
        """Print configuration information"""
        print("\n" + "="*50)
        print("KFilterNet_stepInput Configuration:")
        print("="*50)
        print(f"  State dimension: {self.state_dim}")
        print(f"  Observation dimension: {self.obs_dim}")
        print(f"  Hidden dimension: {self.hidden_dim}")
        print(f"  K module type: {self.K_type}")
        print(f"  H module type: {self.H_type}")
        print(f"  F module type: {self.F_type}")
        print(f"  Q module type: {self.Q_type}")
        print(f"  R module type: {self.R_type}")
        print(f"  Initial state type: {self.init_type}")
        print("="*50 + "\n")

    def _create_fixed_observation_matrix(self):
        """Create fixed observation matrix"""
        H = torch.zeros(self.obs_dim, self.state_dim)
        H[0:3, 0:3] = torch.eye(3)
        return H

    def _create_fixed_transition_matrix(self):
        """Create fixed state transition matrix"""
        F = torch.eye(self.state_dim)
        dt = 0.1
        F[0:3, 3:6] = torch.eye(3) * dt
        F[0:3, 6:9] = 0.5 * torch.eye(3) * dt ** 2
        F[3:6, 6:9] = torch.eye(3) * dt
        return F

    def forward(self, observations, initial_obs=None):
        """Forward propagation"""
        batch_size, seq_len, _ = observations.shape

        # Compute initial state
        state_pred, cov_pred = self._compute_initial_state(observations)

        # Store results
        states = torch.zeros(batch_size, seq_len, self.state_dim, device=observations.device)
        covariances = torch.zeros(batch_size, seq_len, self.state_dim, self.state_dim, device=observations.device)
        kalman_gains = torch.zeros(batch_size, seq_len, self.state_dim, self.obs_dim, device=observations.device)

        # Store intermediate states for feature computation
        prior_states = torch.zeros(batch_size, seq_len, self.state_dim, device=observations.device)
        posterior_states = torch.zeros(batch_size, seq_len, self.state_dim, device=observations.device)
        predicted_obs = torch.zeros(batch_size, seq_len, self.obs_dim, device=observations.device)

        # Store matrices
        H_all = torch.zeros(batch_size, seq_len, self.obs_dim, self.state_dim, device=observations.device)
        F_all = torch.zeros(batch_size, seq_len, self.state_dim, self.state_dim, device=observations.device)
        Q_all = torch.zeros(batch_size, seq_len, self.state_dim, self.state_dim, device=observations.device)
        R_all = torch.zeros(batch_size, seq_len, self.obs_dim, self.obs_dim, device=observations.device)

        prev_obs = observations[:, 0]  # Initialize with first observation
        H_t = self._create_fixed_observation_matrix().unsqueeze(0).expand(batch_size, -1, -1)

        for t in range(seq_len):
            obs_t = observations[:, t]

            # Predict observation ŷ_{t|t-1} = H * x̂_{t|t-1}
            pred_obs_t = torch.bmm(H_t, state_pred.unsqueeze(2)).squeeze(2)
            predicted_obs[:, t] = pred_obs_t

            features = self._compute_features(
                obs_t, prev_obs, pred_obs_t,
                state_pred, states, t, batch_size
            )

            # Process features
            hist_encoded = self.feature_processor(features)

            H_t = self._predict_H_at_t(hist_encoded, batch_size)
            H_all[:, t] = H_t
            R_t = self._predict_R_at_t(hist_encoded, batch_size)
            R_all[:, t] = R_t

            # Predict Kalman gain
            if self.use_learned_K:
                K_t = self._predict_K_at_t(hist_encoded, batch_size)
            else:
                K_t = self._compute_kalman_gain(cov_pred, H_t, R_t)

            kalman_gains[:, t] = K_t

            # Update step
            innovation = obs_t.unsqueeze(2) - pred_obs_t.unsqueeze(2)
            state_updated = state_pred + torch.bmm(K_t, innovation).squeeze(2)

            # Covariance update
            I_KH = torch.eye(self.state_dim, device=observations.device).unsqueeze(0) - torch.bmm(K_t, H_t)
            cov_updated = torch.bmm(torch.bmm(I_KH, cov_pred), I_KH.transpose(1, 2)) + \
                         torch.bmm(torch.bmm(K_t, R_t), K_t.transpose(1, 2))

            # Store results
            states[:, t] = state_updated
            covariances[:, t] = cov_updated
            prior_states[:, t] = state_pred
            posterior_states[:, t] = state_updated

            # Prediction step (prepare for next time step)
            F_t = self._predict_F_at_t(hist_encoded, batch_size)
            F_all[:, t] = F_t
            Q_t = self._predict_Q_at_t(hist_encoded, batch_size)
            Q_all[:, t] = Q_t

            state_pred = torch.bmm(F_t, state_updated.unsqueeze(2)).squeeze(2)
            cov_pred = torch.bmm(torch.bmm(F_t, cov_updated), F_t.transpose(1, 2)) + Q_t

            # Update previous observation
            prev_obs = obs_t

        # Collect intermediate information
        info = {
            'kalman_gains': kalman_gains,
            'observation_matrices': H_all,
            'transition_matrices': F_all,
            'process_noise': Q_all,
            'observation_noise': R_all,
            'prior_states': prior_states,
            'posterior_states': posterior_states,
            'predicted_obs': predicted_obs
        }

        return states, covariances, info

    def _compute_features(self, obs_t, prev_obs, pred_obs_t,
                               state_pred, states, t, batch_size):
        """
        Compute input features F1, F2, F4 from the paper
        (Using the paper-recommended combination {F1, F2, F4})
        """
        # F1: Observation difference Δỹ_t = y_t - y_{t-1}
        f1 = obs_t - prev_obs  # [B, obs_dim]

        # F2: Innovation difference Δy_t = y_t - ŷ_{t|t-1}
        f2 = obs_t - pred_obs_t  # [B, obs_dim]

        # F4: Forward update difference Δx̂_t = x̂_{t|t} - x̂_{t|t-1}
        if t == 0:
            f4 = torch.zeros(batch_size, self.state_dim, device=obs_t.device)
        else:
            # x̂_{t|t} (current posterior) - x̂_{t|t-1} (current prior)
            f4 = states[:, t-1] - state_pred  # [B, state_dim]

        # Combine features
        features = torch.cat([f1, f2, f4], dim=-1)  # [B, obs_dim + obs_dim + state_dim]

        return features

    def _predict_H_at_t(self, hist_encoded, batch_size):
        """Predict observation matrix H at time t"""
        # hist_encoded: [B, hidden_dim]
        if self.H_type == ModuleType.LEARNABLE.value:
            # Expand dimensions to match network input
            hist_encoded = hist_encoded.unsqueeze(1)  # [B, 1, hidden_dim]
            H = self.H_net(hist_encoded).squeeze(1)  # [B, obs_dim, state_dim]
            return H
        elif self.H_type == ModuleType.FIXED.value:
            return self.H_base.unsqueeze(0).expand(batch_size, -1, -1)
        else:  # SEMI_FIXED
            hist_encoded = hist_encoded.unsqueeze(1)  # [B, 1, hidden_dim]
            H_delta = self.H_net(hist_encoded).view(batch_size, self.obs_dim, self.state_dim)
            H_base = self.H_base.unsqueeze(0).expand(batch_size, -1, -1)
            return H_base + H_delta

    def _predict_K_at_t(self, hist_encoded, batch_size):
        """Predict Kalman gain K at time t"""
        # hist_encoded: [B, hidden_dim]
        # Expand dimensions to match network input
        hist_encoded = hist_encoded.unsqueeze(1)  # [B, 1, hidden_dim]
        K = self.K_net(hist_encoded).squeeze(1)  # [B, state_dim, obs_dim]
        return K

    def _predict_F_at_t(self, hist_encoded, batch_size):
        """Predict state transition matrix F at time t"""
        # hist_encoded: [B, hidden_dim]
        if self.F_type == ModuleType.LEARNABLE.value:
            hist_encoded = hist_encoded.unsqueeze(1)  # [B, 1, hidden_dim]
            F = self.F_net(hist_encoded).squeeze(1)
            return F
        elif self.F_type == ModuleType.FIXED.value:
            return self.F_base.unsqueeze(0).expand(batch_size, -1, -1)
        else:  # SEMI_FIXED
            hist_encoded = hist_encoded.unsqueeze(1)  # [B, 1, hidden_dim]
            F_delta = self.F_net(hist_encoded).squeeze(1)
            F_base = self.F_base.unsqueeze(0).expand(batch_size, -1, -1)
            return F_base + F_delta

    def _predict_Q_at_t(self, hist_encoded, batch_size):
        """Predict process noise Q at time t"""
        # hist_encoded: [B, hidden_dim]
        if self.Q_type == ModuleType.LEARNABLE.value:
            hist_encoded = hist_encoded.unsqueeze(1)  # [B, 1, hidden_dim]
            Q_diag = self.Q_net(hist_encoded).squeeze(1)  # [B, state_dim]
            return torch.diag_embed(Q_diag)
        else:
            return self.Q_fixed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).squeeze(1)

    def _predict_R_at_t(self, hist_encoded, batch_size):
        """Predict observation noise R at time t"""
        # hist_encoded: [B, hidden_dim]
        if self.R_type == ModuleType.LEARNABLE.value:
            hist_encoded = hist_encoded.unsqueeze(1)  # [B, 1, hidden_dim]
            R_diag = self.R_net(hist_encoded).squeeze(1)  # [B, obs_dim]
            return torch.diag_embed(R_diag)
        else:
            return self.R_fixed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).squeeze(1)

    def _compute_initial_state(self, observations):
        """Compute initial state"""
        if self.init_type == ModuleType.LEARNABLE.value:
            initial_obs = observations[:, 0]
            state_pred = self.initial_state_net(initial_obs)
            cov_pred = self.initial_cov_net(initial_obs)
            return state_pred, cov_pred
        else:
            if observations.shape[1] >= 3:
                initial_obs_seq = observations[:, :3, :]
                return self.initial_state_computer(initial_obs_seq)
            else:
                initial_obs = observations[:, 0]
                state_pred = torch.zeros(initial_obs.shape[0], self.state_dim, device=observations.device)
                cov_pred = torch.eye(self.state_dim, device=observations.device).unsqueeze(0).expand(initial_obs.shape[0], -1, -1) * 0.1
                return state_pred, cov_pred

    def _compute_kalman_gain(self, cov_pred, H_t, R_t):
        """Mathematical computation of Kalman gain (numerically stable version)"""
        HPHR = torch.bmm(torch.bmm(H_t, cov_pred), H_t.transpose(1, 2)) + R_t

        epsilon = 1e-4
        HPHR_reg = HPHR + epsilon * torch.eye(self.obs_dim, device=H_t.device).unsqueeze(0)

        try:
            K_t = torch.linalg.solve(
                HPHR_reg.transpose(-2, -1),
                torch.bmm(cov_pred, H_t.transpose(1, 2)).transpose(-2, -1)
            ).transpose(-2, -1)
        except:
            K_t = torch.bmm(torch.bmm(cov_pred, H_t.transpose(1, 2)), torch.linalg.pinv(HPHR_reg))

        return K_t
