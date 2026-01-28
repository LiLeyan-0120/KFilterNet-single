"""
Centralized constants for the Kalman Filter and Deep Learning Fusion project.

This module contains all magic numbers, thresholds, and configuration values
used throughout the codebase to improve code maintainability and consistency.
"""

from enum import Enum

# ============================================================================
# TIME CONSTANTS
# ============================================================================

DEFAULT_DT = 0.02
DEFAULT_SEQUENCE_LENGTH = 250

# ============================================================================
# NUMERICAL STABILITY
# ============================================================================

STABILITY_EPSILON = 1e-6
NUMERICAL_EPSILON = 1e-10
KALMAN_GAIN_REGULARIZATION = 1e-4

# ============================================================================
# VALUE CLAMPING
# ============================================================================

LOG_SOFTPLUS_MIN = -5.0
LOG_SOFTPLUS_MAX = 5.0
CORRELATION_COEFF_MIN = -10.0
CORRELATION_COEFF_MAX = 10.0
COVARIANCE_CLAMP_MIN = -1000.0
COVARIANCE_CLAMP_MAX = 1000.0

# ============================================================================
# ADDITIONAL CONSTANTS
# ============================================================================

DEFAULT_TURN_RATE = 0.2  # rad/s (~11.5 deg/s)
MAX_STATE_CHANGE_THRESHOLD = 1000.0
INNOVATION_TERM_THRESHOLD = 50.0
LOG_LIKELIHOOD_CLAMP_MIN = -50.0
LOG_LIKELIHOOD_CLAMP_MAX = 50.0
LIKELIHOOD_MIN = 1e-15
LIKELIHOOD_MAX = 1e10

# ============================================================================
# KINEMATIC CONSTRAINTS
# ============================================================================

POSITION_RANGE = (-1000.0, 1000.0)
DEFAULT_VELOCITY_MIN = 100.0
DEFAULT_VELOCITY_MAX = 500.0
DEFAULT_ACCELERATION = 12.0
MIN_FLIGHT_SPEED_RATIO = 0.2
CLIMB_ANGLE_RANGE = (30.0, 80.0)
DIVE_ANGLE_RANGE = (-60.0, -30.0)
LOOP_RADIUS_RANGE = (500.0, 1000.0)

# ============================================================================
# TURN MOTION
# ============================================================================

MAX_TURN_RATE_DEGREES = 30.0
WEAVING_FREQUENCY_RANGE = (0.2, 0.5)
WEAVING_AMPLITUDE_RANGE = (0.3, 0.6)
MIN_MODE_DURATION = 2.0

# ============================================================================
# GRAVITY
# ============================================================================

GRAVITY = 9.8

# ============================================================================
# NOISE PARAMETERS
# ============================================================================

IMPULSE_PROBABILITY = 0.05
IMPULSE_MAGNITUDE_RANGE = (5.0, 20.0)
NOISE_CORRELATION_COEFFICIENT = 0.4
OBSERVATION_NOISE_DEFAULT_STD = 0.5
PROCESS_NOISE_DEFAULT_STD = 0.1


class ModuleType(Enum):
    """Module type enumeration for Kalman filter components.

    Options:
        LEARNABLE: Full network method (BaseKalmanNet style) - all parameters learned
        FIXED: Fixed matrix method (TrackFusionNet style) - classical matrix operations
        SEMI_FIXED: Semi-fixed method - base matrix + residual learning
        HYBRID: Hybrid method - network prediction with mathematical constraints
    """
    LEARNABLE = "learnable"
    FIXED = "fixed"
    SEMI_FIXED = "semi_fixed"
    HYBRID = "hybrid"