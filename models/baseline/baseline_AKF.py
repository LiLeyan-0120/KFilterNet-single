import numpy as np


class BaselineAdaptiveKF:
    """Adaptive Kalman Filter"""

    def __init__(self, state_dim=9, obs_dim=3, dt=0.02):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt

        # State transition matrix
        self.F = self._create_constant_acceleration_matrix()

        # Observation matrix
        self.H = np.zeros((obs_dim, state_dim))
        self.H[0:3, 0:3] = np.eye(3)

        # Adaptive parameters
        self.alpha = 0.95  # Forgetting factor
        self.window_size = 10  # Sliding window size

    def _create_constant_acceleration_matrix(self):
        """Create state transition matrix for constant acceleration motion"""
        F = np.eye(self.state_dim)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[0:3, 6:9] = 0.5 * np.eye(3) * self.dt ** 2
        F[3:6, 6:9] = np.eye(3) * self.dt
        return F

    def _estimate_noise(self, innovations):
        """Estimate noise covariance from innovation sequence"""
        if len(innovations) < 2:
            return np.eye(self.obs_dim) * 0.5

        innovations = np.array(innovations)
        R_est = np.cov(innovations.T)

        # Ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(R_est)
        eigenvals = np.maximum(eigenvals, 1e-6)
        R_est = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return R_est

    def filter(self, observations):
        """Adaptive Kalman filtering"""
        T = observations.shape[0]
        states = np.zeros((T, self.state_dim))

        # Initialization
        state = np.zeros(self.state_dim)
        state[0:3] = observations[0]
        state[3:6] = (observations[1] - observations[0]) / self.dt if T > 1 else np.zeros(3)
        state[6:9] = np.zeros(3)

        P = np.eye(self.state_dim) * 1.0
        Q = np.eye(self.state_dim) * 0.1
        R = np.eye(self.obs_dim) * 0.5

        innovations = []

        for t in range(T):
            # Prediction
            state_pred = self.F @ state
            P_pred = self.F @ P @ self.F.T + Q

            # Update
            obs = observations[t]
            innovation = obs - self.H @ state_pred
            innovations.append(innovation)

            # Adaptive adjustment of R
            if len(innovations) >= self.window_size:
                recent_innovations = innovations[-self.window_size:]
                R = self._estimate_noise(recent_innovations)

            S = self.H @ P_pred @ self.H.T + R
            try:
                K = P_pred @ self.H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
                K = P_pred @ self.H.T @ S_inv

            state = state_pred + K @ innovation
            P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

            states[t] = state

        return states
