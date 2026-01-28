import numpy as np

class BaselineKalmanFilter:
    """Standard Kalman Filter"""

    def __init__(self, state_dim=9, obs_dim=3, dt=0.02):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt

        # Use constant acceleration model (can degrade to CV when acceleration is zero)
        self.F = self._create_constant_acceleration_matrix()

        # Fixed observation matrix
        self.H = np.zeros((obs_dim, state_dim))
        self.H[0:3, 0:3] = np.eye(3)

        # Initialize covariance matrices based on data generation parameters
        # Process noise: based on process_noise_range (0.01, 0.5), use moderate value
        self.Q = np.eye(state_dim) * 0.1
        # Observation noise: based on observation_noise_range (0.1, 3.0), use moderate value
        self.R = np.eye(obs_dim) * 0.5

    def _create_constant_acceleration_matrix(self):
        """Create state transition matrix for constant acceleration motion"""
        F = np.eye(self.state_dim)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[0:3, 6:9] = 0.5 * np.eye(3) * self.dt ** 2
        F[3:6, 6:9] = np.eye(3) * self.dt
        return F

    def filter(self, observations):
        """Kalman filtering"""
        T = observations.shape[0]
        states = np.zeros((T, self.state_dim))

        # Initial state
        state = np.zeros(self.state_dim)
        state[0:3] = observations[0]  # Initialize position with first observation
        state[3:6] = (observations[1] - observations[0]) / self.dt if T > 1 else np.zeros(3)  # Initial velocity
        state[6:9] = np.zeros(3)  # Initial acceleration

        P = np.eye(self.state_dim) * 1.0  # Initial covariance

        for t in range(T):
            # Prediction step
            state_pred = self.F @ state
            P_pred = self.F @ P @ self.F.T + self.Q

            # Update step
            obs = observations[t]
            innovation = obs - self.H @ state_pred
            S = self.H @ P_pred @ self.H.T + self.R

            # Kalman gain
            try:
                K = P_pred @ self.H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = P_pred @ self.H.T @ np.linalg.pinv(S)

            # State update
            state = state_pred + K @ innovation
            P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

            states[t] = state

        return states
