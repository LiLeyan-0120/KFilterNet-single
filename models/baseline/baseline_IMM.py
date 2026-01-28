import numpy as np

class BaselineIMM:
    """Classic Interacting Multiple Model Filter (IMM)
    
    Implemented according to the classic IMM algorithm by Blom and Bar-Shalom:
    1. Model condition reinitialization (mixing)
    2. Model condition filtering
    3. Model probability update
    4. State and covariance estimation fusion
    """

    def __init__(self, state_dim=9, obs_dim=3, dt=0.02):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        
        # Classic 3-model IMM architecture: CV, CA, CT
        self.num_models = 3
        
        # CT model turn rate: 0.2 rad/s (~11.5 deg/s)
        # This is a moderate turn rate that can handle most turning scenarios
        # from the data generator (-30 deg/s to 30 deg/s)
        self.omega = 0.2

        # Three models: CV, CA, CT
        self.model_names = ['CV', 'CA', 'CT']
        self.models = {
            'CV': self._create_cv_model(),
            'CA': self._create_ca_model(),
            'CT': self._create_ct_model(omega=self.omega)
        }

        # Model transition probability matrix (Markov chain)
        self.transition_prob_matrix = np.array([
            [0.7, 0.2, 0.1],  # CV -> [CV, CA, CT]
            [0.3, 0.5, 0.2],  # CA -> [CV, CA, CT]
            [0.2, 0.1, 0.7]   # CT -> [CV, CA, CT]
        ])

        # Initial model probabilities
        self.model_probabilities = np.array([1/3, 1/3, 1/3])

        # Observation matrix
        self.H = np.zeros((obs_dim, state_dim))
        self.H[0:3, 0:3] = np.eye(3)
        
        # Observation noise matrix based on data generation parameters
        self.R = np.eye(obs_dim) * 0.5  # Based on observation_noise_range (0.1, 3.0)

    def _create_cv_model(self):
        """Create Constant Velocity (CV) model"""
        F = np.eye(self.state_dim)
        F[0:3, 3:6] = np.eye(3) * self.dt
        
        # CV model process noise: based on process_noise_range (0.01, 0.5)
        # Use moderate value for CV model
        q = 0.01  # Process noise intensity
        Q = np.zeros((self.state_dim, self.state_dim))
        Q[0:3, 0:3] = q * self.dt**3 / 3 * np.eye(3)
        Q[0:3, 3:6] = q * self.dt**2 / 2 * np.eye(3)
        Q[3:6, 0:3] = q * self.dt**2 / 2 * np.eye(3)
        Q[3:6, 3:6] = q * self.dt * np.eye(3)
        Q[6:9, 6:9] = 0.001 * np.eye(3)  # Acceleration noise is very small
        
        return {'F': F, 'Q': Q}

    def _create_ca_model(self):
        """Create Constant Acceleration (CA) model"""
        F = np.eye(self.state_dim)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[0:3, 6:9] = 0.5 * np.eye(3) * self.dt ** 2
        F[3:6, 6:9] = np.eye(3) * self.dt
        
        # CA model process noise: based on process_noise_range (0.01, 0.5)
        # Use moderate value for CA model
        q = 0.1  # Process noise intensity
        Q = np.zeros((self.state_dim, self.state_dim))
        Q[0:3, 0:3] = q * self.dt**5 / 20 * np.eye(3)
        Q[0:3, 3:6] = q * self.dt**4 / 8 * np.eye(3)
        Q[0:3, 6:9] = q * self.dt**3 / 6 * np.eye(3)
        Q[3:6, 0:3] = q * self.dt**4 / 8 * np.eye(3)
        Q[3:6, 3:6] = q * self.dt**3 / 3 * np.eye(3)
        Q[3:6, 6:9] = q * self.dt**2 / 2 * np.eye(3)
        Q[6:9, 0:3] = q * self.dt**3 / 6 * np.eye(3)
        Q[6:9, 3:6] = q * self.dt**2 / 2 * np.eye(3)
        Q[6:9, 6:9] = q * self.dt * np.eye(3)
        
        return {'F': F, 'Q': Q}

    def _create_ct_model(self, omega=0.1):
        """Create Coordinated Turn (CT) model with specified turn rate"""
        F = np.eye(self.state_dim)
        
        dt = self.dt
        
        if abs(omega) > 1e-6:  # Has turn
            # Coordinated turn state transition in XY plane - Corrected
            sin_wt = np.sin(omega * dt)
            cos_wt = np.cos(omega * dt)

            # Corrected XY plane coordinated turn model
            F[0, 3] = (cos_wt - 1) / omega
            F[0, 4] = sin_wt / omega
            F[1, 3] = -sin_wt / omega
            F[1, 4] = (cos_wt - 1) / omega
            F[3, 3] = cos_wt
            F[3, 4] = sin_wt
            F[4, 3] = -sin_wt
            F[4, 4] = cos_wt
        else:  # Straight motion, degrades to CV model
            F[0:3, 3:6] = np.eye(3) * dt
        
        # Z direction maintains constant velocity
        F[2, 5] = dt
        
        # CT model process noise: based on process_noise_range (0.01, 0.5)
        # Use moderate value for CT model
        q = 0.05  # Process noise intensity
        Q = np.zeros((self.state_dim, self.state_dim))
        Q[0:3, 0:3] = q * self.dt**3 / 3 * np.eye(3)
        Q[0:3, 3:6] = q * self.dt**2 / 2 * np.eye(3)
        Q[3:6, 0:3] = q * self.dt**2 / 2 * np.eye(3)
        Q[3:6, 3:6] = q * self.dt * np.eye(3)
        Q[6:9, 6:9] = 0.01 * np.eye(3)  # Acceleration noise during turn
        
        return {'F': F, 'Q': Q}

    def _compute_mixing_probabilities(self):
        """Compute mixing probabilities (Classic IMM step 1)"""
        # c_j = sum_i(transition_prob_matrix_ij * model_probabilities_i)
        c = self.transition_prob_matrix.T @ self.model_probabilities

        # mixing_probabilities_ij = (transition_prob_matrix_ij * model_probabilities_i) / c_j
        mixing_probabilities = np.zeros((self.num_models, self.num_models))
        for i in range(self.num_models):
            for j in range(self.num_models):
                if c[j] > 1e-10:
                    mixing_probabilities[i, j] = (self.transition_prob_matrix[i, j] * self.model_probabilities[i]) / c[j]
                else:
                    mixing_probabilities[i, j] = 1.0 / self.num_models  # Uniform distribution

        return mixing_probabilities, c

    def _mix_states(self, mixing_probabilities, states, covs):
        """Mix states and covariances (Classic IMM step 2)"""
        mixed_states = []
        mixed_covs = []

        for j in range(self.num_models):
            # Compute mixed state
            mixed_state = np.zeros(self.state_dim)
            for i in range(self.num_models):
                mixed_state += mixing_probabilities[i, j] * states[i]

            # Compute mixed covariance
            mixed_cov = np.zeros((self.state_dim, self.state_dim))
            for i in range(self.num_models):
                diff = states[i] - mixed_state
                mixed_cov += mixing_probabilities[i, j] * (covs[i] + np.outer(diff, diff))

            mixed_states.append(mixed_state)
            mixed_covs.append(mixed_cov)

        return mixed_states, mixed_covs

    def _kalman_filter(self, state, P, F, Q, observation):
        """Standard Kalman filter step"""
        # Prediction
        state_pred = F @ state
        P_pred = F @ P @ F.T + Q
        
        # Update
        innovation = observation - self.H @ state_pred
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Numerical stability handling
        try:
            S_inv = np.linalg.inv(S)
            K = P_pred @ self.H.T @ S_inv
        except np.linalg.LinAlgError:
            # Use pseudo-inverse or simplified handling
            S_inv = np.linalg.pinv(S)
            K = P_pred @ self.H.T @ S_inv
        
        state_update = state_pred + K @ innovation
        P_update = (np.eye(self.state_dim) - K @ self.H) @ P_pred
        
        # Compute likelihood - improved numerical stability handling
        try:
            det_S = np.linalg.det(S)
            if det_S <= 1e-10:
                det_S = 1e-10
            
            innovation_term = innovation.T @ S_inv @ innovation
            
            # Stricter numerical stability check
            if innovation_term > 50:  # Lower threshold to avoid exp overflow
                likelihood = 1e-15  # Use smaller value to avoid underflow
            elif innovation_term < 0:  # Check for invalid values
                likelihood = 1e-15
            else:
                # Use log-space computation to avoid overflow
                log_likelihood = -0.5 * innovation_term - 0.5 * np.log(det_S * (2 * np.pi) ** self.obs_dim)
                
                # Limit log_likelihood range
                log_likelihood = np.clip(log_likelihood, -50, 50)
                
                # Convert from log-space back to linear-space
                likelihood = np.exp(log_likelihood)
                
                # Ensure likelihood is in reasonable range
                likelihood = np.clip(likelihood, 1e-15, 1e10)
                
        except Exception as e:
            # More detailed error handling
            likelihood = 1e-15
            if np.any(np.isnan(innovation)) or np.any(np.isinf(innovation)):
                likelihood = 1e-15
            if np.any(np.isnan(S_inv)) or np.any(np.isinf(S_inv)):
                likelihood = 1e-15
        
        return state_update, P_update, likelihood

    def filter(self, observations):
        """Classic IMM filtering algorithm"""
        T = observations.shape[0]
        states = np.zeros((T, self.state_dim))

        # Initialize state and covariance for each model
        model_states = []
        model_covs = []
        for i in range(self.num_models):
            state = np.zeros(self.state_dim)
            state[0:3] = observations[0]
            if T > 1:
                state[3:6] = (observations[1] - observations[0]) / self.dt
                if T > 2:
                    state[6:9] = (observations[2] - 2*observations[1] + observations[0]) / (self.dt ** 2)
            model_states.append(state)
            model_covs.append(np.eye(self.state_dim) * 1.0)

        for t in range(T):
            # Step 1: Compute mixing probabilities
            mixing_probabilities, c = self._compute_mixing_probabilities()

            # Step 2: Mix states and covariances
            mixed_states, mixed_covs = self._mix_states(mixing_probabilities, model_states, model_covs)

            # Step 3: Model condition filtering
            likelihoods = []
            updated_states = []
            updated_covs = []

            for i in range(self.num_models):
                model_name = self.model_names[i]
                model = self.models[model_name]

                state_update, P_update, likelihood = self._kalman_filter(
                    mixed_states[i], mixed_covs[i],
                    model['F'], model['Q'], observations[t]
                )

                updated_states.append(state_update)
                updated_covs.append(P_update)
                likelihoods.append(likelihood)

            # Step 4: Model probability update - improved numerical stability handling
            likelihoods = np.array(likelihoods)

            # Check validity of likelihoods
            if np.any(np.isnan(likelihoods)) or np.any(np.isinf(likelihoods)):
                # If there are invalid values, reset to uniform distribution
                self.model_probabilities = np.ones(self.num_models) / self.num_models
            else:
                # Ensure all likelihoods are positive
                likelihoods = np.maximum(likelihoods, 1e-15)

                # Compute unnormalized model probabilities
                self.model_probabilities = likelihoods * c

                # Check validity of computation results
                mu_sum = self.model_probabilities.sum()
                if np.isnan(mu_sum) or np.isinf(mu_sum) or mu_sum <= 1e-15:
                    # If result is invalid, reset to uniform distribution
                    self.model_probabilities = np.ones(self.num_models) / self.num_models
                else:
                    # Normalize
                    self.model_probabilities = self.model_probabilities / mu_sum

                    # Final check to ensure probabilities are valid
                    if np.any(np.isnan(self.model_probabilities)) or np.any(np.isinf(self.model_probabilities)):
                        self.model_probabilities = np.ones(self.num_models) / self.num_models
            
            # Step 5: State and covariance fusion - add numerical stability check
            output_state = np.zeros(self.state_dim)
            for i in range(self.num_models):
                # Check validity of state
                if np.any(np.isnan(updated_states[i])) or np.any(np.isinf(updated_states[i])):
                    # If state is invalid, use previous valid state or zero state
                    if t > 0 and not np.any(np.isnan(states[t-1])) and not np.any(np.isinf(states[t-1])):
                        state_to_use = states[t-1]
                    else:
                        state_to_use = np.zeros(self.state_dim)
                        state_to_use[0:3] = observations[t]  # At least use observation
                else:
                    state_to_use = updated_states[i]

                output_state += self.model_probabilities[i] * state_to_use
            
            # Final check validity of output state
            if np.any(np.isnan(output_state)) or np.any(np.isinf(output_state)):
                # If output is invalid, use observation or previous state
                if t > 0 and not np.any(np.isnan(states[t-1])) and not np.any(np.isinf(states[t-1])):
                    output_state = states[t-1]
                else:
                    output_state = np.zeros(self.state_dim)
                    output_state[0:3] = observations[t]
            
            # Update model states - add validity check
            for i in range(self.num_models):
                if np.any(np.isnan(updated_states[i])) or np.any(np.isinf(updated_states[i])):
                    # If state is invalid, use mixed state or previous state
                    if t > 0 and i < len(model_states):
                        updated_states[i] = model_states[i]
                    else:
                        updated_states[i] = np.zeros(self.state_dim)
                        updated_states[i][0:3] = observations[t]
                
                if np.any(np.isnan(updated_covs[i])) or np.any(np.isinf(updated_covs[i])):
                    # If covariance is invalid, reset to identity matrix
                    updated_covs[i] = np.eye(self.state_dim) * 1.0
            
            model_states = updated_states
            model_covs = updated_covs
            
            states[t] = output_state

        return states
