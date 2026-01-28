import numpy as np


class BaselineUKF:
    """
    Unscented Kalman Filter (UKF)
    A nonlinear filter that uses the unscented transform to handle nonlinear systems
    """

    def __init__(self, state_dim=9, obs_dim=3, dt=0.02, alpha=1e-3, beta=2.0, kappa=0.0):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt

        # UKF parameters
        self.alpha = alpha  # Spread of sigma points
        self.beta = beta    # Prior knowledge of distribution (2 for Gaussian)
        self.kappa = kappa  # Secondary scaling parameter

        # Calculate lambda
        self.lambda_ = self.alpha ** 2 * (self.state_dim + self.kappa) - self.state_dim

        # Calculate weights
        self.Wm = np.zeros(2 * self.state_dim + 1)  # Weights for mean
        self.Wc = np.zeros(2 * self.state_dim + 1)  # Weights for covariance

        self.Wm[0] = self.lambda_ / (self.state_dim + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.state_dim + self.lambda_) + (1 - self.alpha ** 2 + self.beta)

        for i in range(1, 2 * self.state_dim + 1):
            self.Wm[i] = 1.0 / (2 * (self.state_dim + self.lambda_))
            self.Wc[i] = 1.0 / (2 * (self.state_dim + self.lambda_))

        # Use constant acceleration model (can degrade to CV when acceleration is zero)
        self.F = self._create_constant_acceleration_matrix()

        # Observation matrix
        self.H = np.zeros((obs_dim, state_dim))
        self.H[0:3, 0:3] = np.eye(3)

        # Process and observation noise covariances based on data generation parameters
        self.Q = np.eye(state_dim) * 0.1  # Based on process_noise_range (0.01, 0.5)
        self.R = np.eye(obs_dim) * 0.5    # Based on observation_noise_range (0.1, 3.0)

    def _create_constant_acceleration_matrix(self):
        """Create state transition matrix for constant acceleration motion"""
        F = np.eye(self.state_dim)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[0:3, 6:9] = 0.5 * np.eye(3) * self.dt ** 2
        F[3:6, 6:9] = np.eye(3) * self.dt
        return F

    def _generate_sigma_points(self, x, P):
        """Generate sigma points"""
        sigma_points = np.zeros((2 * self.state_dim + 1, self.state_dim))
        sigma_points[0] = x

        # Cholesky decomposition
        try:
            sqrt_P = np.linalg.cholesky((self.state_dim + self.lambda_) * P).T
        except np.linalg.LinAlgError:
            # Add small regularization if matrix is not positive definite
            sqrt_P = np.linalg.cholesky((self.state_dim + self.lambda_) * P + 1e-6 * np.eye(self.state_dim)).T

        for i in range(self.state_dim):
            sigma_points[i + 1] = x + sqrt_P[i]
            sigma_points[self.state_dim + i + 1] = x - sqrt_P[i]

        return sigma_points

    def _unscented_transform(self, sigma_points, noise_cov=None):
        """Apply unscented transform to sigma points"""
        # Calculate mean
        mean = np.sum(self.Wm[:, np.newaxis] * sigma_points, axis=0)

        # Calculate covariance
        diff = sigma_points - mean
        cov = np.sum(self.Wc[:, np.newaxis, np.newaxis] * diff[:, :, np.newaxis] * diff[:, np.newaxis, :], axis=0)

        # Add noise covariance if provided
        if noise_cov is not None:
            cov += noise_cov

        return mean, cov

    def _state_transition(self, sigma_points):
        """Apply state transition function to sigma points"""
        # Linear state transition: x = F * x
        transformed = sigma_points @ self.F.T
        return transformed

    def _observation_function(self, sigma_points):
        """Apply observation function to sigma points"""
        # Linear observation: y = H * x
        transformed = sigma_points @ self.H.T
        return transformed

    def filter(self, observations):
        """Unscented Kalman filtering"""
        T = observations.shape[0]
        states = np.zeros((T, self.state_dim))

        # Initialize state
        x = np.zeros(self.state_dim)
        x[0:3] = observations[0]
        x[3:6] = (observations[1] - observations[0]) / self.dt if T > 1 else np.zeros(3)
        x[6:9] = np.zeros(3)

        # Initialize covariance
        P = np.eye(self.state_dim) * 1.0

        for t in range(T):
            # Generate sigma points
            sigma_points = self._generate_sigma_points(x, P)

            # Predict step
            # Transform sigma points through state transition
            sigma_points_pred = self._state_transition(sigma_points)

            # Predicted state and covariance
            x_pred, P_pred = self._unscented_transform(sigma_points_pred, self.Q)

            # Transform sigma points through observation function
            sigma_points_obs = self._observation_function(sigma_points_pred)

            # Predicted observation and covariance
            y_pred, Pyy = self._unscented_transform(sigma_points_obs, self.R)

            # Cross covariance
            diff_x = sigma_points_pred - x_pred
            diff_y = sigma_points_obs - y_pred
            Pxy = np.sum(self.Wc[:, np.newaxis, np.newaxis] * diff_x[:, :, np.newaxis] * diff_y[:, np.newaxis, :], axis=0)

            # Kalman gain
            try:
                K = Pxy @ np.linalg.inv(Pyy)
            except np.linalg.LinAlgError:
                K = Pxy @ np.linalg.pinv(Pyy)

            # Update step
            obs = observations[t]
            innovation = obs - y_pred

            x = x_pred + K @ innovation
            P = P_pred - K @ Pyy @ K.T

            states[t] = x

        return states
