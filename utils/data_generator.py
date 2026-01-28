import numpy as np
import torch
from typing import Tuple, List, Optional, Dict, Any
import os
import sys
import pickle
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from enum import Enum
import logging

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MotionMode(Enum):
    """Motion mode enumeration"""
    CONSTANT_VELOCITY = 0
    CONSTANT_ACCELERATION = 1
    COORDINATED_TURN = 2
    WEAVING = 3
    VERTICAL_MANEUVER = 4
    LOOP = 5
    ZOOM_CLIMB = 6
    DIVE_ATTACK = 7

@dataclass
class TrajectorySample:
    """Trajectory sample class"""
    states: np.ndarray  # [T, 9] - [x,y,z,vx,vy,vz,ax,ay,az]
    observations: np.ndarray  # [T, 3] - [x,y,z]
    process_noise: float
    observation_noise: float
    motion_type: str
    dt: float = 0.1  # Time step
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to PyTorch tensors"""
        return torch.FloatTensor(self.states), torch.FloatTensor(self.observations)
    
    def validate(self) -> bool:
        """Validate physical reasonableness of sample"""
        # Check data shape
        if self.states.shape[1] != 9 or self.observations.shape[1] != 3:
            return False
        
        # Check time step consistency
        if len(self.states) != len(self.observations):
            return False
        
        # Check numerical validity
        if not np.all(np.isfinite(self.states)) or not np.all(np.isfinite(self.observations)):
            return False
        
        return True

class TrajectoryGenerator:
    """Improved trajectory data generator"""
    
    def __init__(self, config):
        self.config = config
        self.state_dim = 9
        self.obs_dim = 3
        self.dt = config.dt if hasattr(config, 'dt') else 0.1
        
        # Kinematic constraints
        self.max_acceleration = config.max_acceleration
        self.max_velocity = config.max_velocity
        
        # Noise generation state for correlated noise
        self._prev_obs_noise = np.zeros(3)
        self._prev_process_noise = np.zeros(self.state_dim)
        
        # Reset noise state for new sample generation
        self._reset_noise_state()
    
    def _reset_noise_state(self):
        """Reset noise state for new trajectory generation"""
        self._prev_obs_noise = np.zeros(3)
        self._prev_process_noise = np.zeros(self.state_dim)
    
    def _generate_impulse_noise(self, base_noise: np.ndarray, noise_std: float) -> np.ndarray:
        """
        Generate impulse noise (electronic warfare interference)
        
        Args:
            base_noise: Base Gaussian noise
            noise_std: Standard deviation of base noise
            
        Returns:
            Noise with potential impulse added
        """
        if not self.config.enable_impulse_noise:
            return base_noise
        
        # Check if impulse occurs
        if np.random.random() < self.config.impulse_probability:
            # Generate impulse magnitude
            impulse_magnitude = np.random.uniform(
                self.config.impulse_magnitude_range[0],
                self.config.impulse_magnitude_range[1]
            )
            
            # Generate random direction for impulse
            impulse_direction = np.random.randn(len(base_noise))
            impulse_direction = impulse_direction / np.linalg.norm(impulse_direction)
            
            # Add impulse to base noise
            impulse = impulse_direction * impulse_magnitude * noise_std
            return base_noise + impulse
        
        return base_noise
    
    def _generate_correlated_noise(self, base_noise: np.ndarray, 
                                   prev_noise: np.ndarray) -> np.ndarray:
        """
        Generate correlated noise using first-order Markov process
        
        Args:
            base_noise: Current white noise sample
            prev_noise: Previous noise sample
            
        Returns:
            Correlated noise
        """
        if not self.config.enable_correlated_noise:
            return base_noise
        
        # First-order Markov process: n_k = rho * n_{k-1} + sqrt(1-rho^2) * w_k
        rho = self.config.noise_correlation_coeff
        correlated_noise = rho * prev_noise + np.sqrt(1 - rho**2) * base_noise
        
        return correlated_noise
    
    def _generate_observation_noise(self, observation_noise_std: float) -> np.ndarray:
        """
        Generate observation noise with impulse and correlation
        
        Args:
            observation_noise_std: Standard deviation of observation noise
            
        Returns:
            Observation noise vector
        """
        # Generate base Gaussian noise
        base_noise = np.random.normal(0, observation_noise_std, 3)
        
        # Add impulse noise
        noise_with_impulse = self._generate_impulse_noise(base_noise, observation_noise_std)
        
        # Add correlation
        correlated_noise = self._generate_correlated_noise(
            noise_with_impulse, 
            self._prev_obs_noise
        )
        
        # Update previous noise state
        self._prev_obs_noise = correlated_noise.copy()
        
        return correlated_noise
    
    def _generate_process_noise_vector(self, process_noise_std: float, 
                                       motion_mode: MotionMode) -> np.ndarray:
        """
        Generate process noise vector with correlation
        
        Args:
            process_noise_std: Standard deviation of process noise
            motion_mode: Current motion mode
            
        Returns:
            Process noise vector
        """
        # Generate base noise based on motion mode
        Q = self._build_process_noise_matrix(motion_mode, process_noise_std)
        base_noise = np.random.multivariate_normal(np.zeros(self.state_dim), Q)
        
        # Add correlation to process noise
        if self.config.enable_correlated_noise:
            rho = self.config.noise_correlation_coeff
            correlated_noise = rho * self._prev_process_noise + np.sqrt(1 - rho**2) * base_noise
            self._prev_process_noise = correlated_noise.copy()
            return correlated_noise
        
        return base_noise

    def _build_state_transition_matrix(self, motion_mode: MotionMode, **kwargs) -> np.ndarray:
        """Build state transition matrix"""
        F = np.eye(self.state_dim)
        
        if motion_mode == MotionMode.CONSTANT_VELOCITY:
            # Constant velocity: x_{k+1} = x_k + v_k * dt
            F[0:3, 3:6] = np.eye(3) * self.dt
            
        elif motion_mode == MotionMode.CONSTANT_ACCELERATION:
            # Constant acceleration: x_{k+1} = x_k + v_k * dt + 0.5 * a_k * dt^2
            # v_{k+1} = v_k + a_k * dt
            F[0:3, 3:6] = np.eye(3) * self.dt
            F[0:3, 6:9] = 0.5 * np.eye(3) * self.dt ** 2
            F[3:6, 6:9] = np.eye(3) * self.dt
            
        elif motion_mode == MotionMode.COORDINATED_TURN:
            # Coordinated turn: using discretized coordinated turn model
            turn_rate = kwargs.get('turn_rate', 0.0)
            if abs(turn_rate) < 1e-6:
                # Approximate as constant velocity
                F[0:3, 3:6] = np.eye(3) * self.dt
            else:
                # Precise coordinated turn state transition
                omega = turn_rate
                sin_omega_dt = np.sin(omega * self.dt)
                cos_omega_dt = np.cos(omega * self.dt)
                
                # X-Y plane coordinated turn
                F[0, 0] = 1.0
                F[0, 1] = sin_omega_dt / omega
                F[0, 3] = (1 - cos_omega_dt) / omega
                F[0, 4] = sin_omega_dt / omega
                
                F[1, 0] = -sin_omega_dt / omega
                F[1, 1] = 1.0
                F[1, 3] = -sin_omega_dt / omega
                F[1, 4] = (1 - cos_omega_dt) / omega
                
                F[3, 0] = 0.0
                F[3, 1] = -omega * sin_omega_dt
                F[3, 3] = cos_omega_dt
                F[3, 4] = sin_omega_dt
                
                F[4, 0] = omega * sin_omega_dt
                F[4, 1] = 0.0
                F[4, 3] = -sin_omega_dt
                F[4, 4] = cos_omega_dt
                
                # Z direction maintains constant velocity
                F[2, 5] = self.dt
                
        return F
    
    def _build_process_noise_matrix(self, motion_mode: MotionMode, process_noise: float) -> np.ndarray:
        """Build process noise matrix"""
        Q = np.zeros((self.state_dim, self.state_dim))
        
        # Basic noise intensity
        q = process_noise
        
        if motion_mode == MotionMode.CONSTANT_VELOCITY:
            # Constant velocity: position and velocity both have noise
            Q[0:3, 0:3] = q * np.eye(3) * self.dt      # Position noise
            Q[3:6, 3:6] = q * np.eye(3) * self.dt * 0.1  # Velocity noise (smaller)
            Q[6:9, 6:9] = q * np.eye(3) * self.dt * 0.01  # Acceleration noise (very small)
            
        elif motion_mode == MotionMode.CONSTANT_ACCELERATION:
            # Constant acceleration: position, velocity, acceleration all have noise
            Q[0:3, 0:3] = q * np.eye(3) * self.dt
            Q[3:6, 3:6] = q * np.eye(3) * self.dt * 0.1
            Q[6:9, 6:9] = q * np.eye(3) * self.dt * 0.05  # Acceleration noise smaller
            
        elif motion_mode == MotionMode.COORDINATED_TURN:
            # Coordinated turn: main noise in position and velocity
            Q[0:3, 0:3] = q * np.eye(3) * self.dt
            Q[3:6, 3:6] = q * np.eye(3) * self.dt * 0.1
            Q[6:9, 6:9] = q * np.eye(3) * self.dt * 0.01
            
        elif motion_mode == MotionMode.WEAVING:
            # Weaving: sinusoidal turn, moderate noise
            Q[0:3, 0:3] = q * np.eye(3) * self.dt
            Q[3:6, 3:6] = q * np.eye(3) * self.dt * 0.15
            Q[6:9, 6:9] = q * np.eye(3) * self.dt * 0.02
            
        elif motion_mode == MotionMode.VERTICAL_MANEUVER:
            # Vertical maneuver (climb or dive): higher noise in Z direction
            Q[0:3, 0:3] = q * np.eye(3) * self.dt
            Q[3:6, 3:6] = q * np.eye(3) * self.dt * 0.1
            Q[6:9, 6:9] = q * np.eye(3) * self.dt * 0.05
            # Higher noise in Z direction
            Q[2, 2] = q * self.dt * 1.5
            Q[5, 5] = q * self.dt * 0.2
            Q[8, 8] = q * self.dt * 0.1
            
        elif motion_mode == MotionMode.LOOP:
            # Loop: vertical coordinated turn, moderate noise
            Q[0:3, 0:3] = q * np.eye(3) * self.dt
            Q[3:6, 3:6] = q * np.eye(3) * self.dt * 0.15
            Q[6:9, 6:9] = q * np.eye(3) * self.dt * 0.03
            
        return Q
    
    def _apply_kinematic_constraints(self, state: np.ndarray) -> np.ndarray:
        """Apply kinematic constraints"""
        # Velocity constraint
        velocity = state[3:6]
        speed = np.linalg.norm(velocity)
        if speed > self.max_velocity:
            state[3:6] = velocity * self.max_velocity / speed
        
        # Acceleration constraint
        acceleration = state[6:9]
        acc_norm = np.linalg.norm(acceleration)
        if acc_norm > self.max_acceleration:
            state[6:9] = acceleration * self.max_acceleration / acc_norm
            
        return state
    
    def _generate_initial_state(self, motion_type: str) -> np.ndarray:
        """Generate reasonable initial state (for aerial targets)"""
        # Position: randomly distributed in large range
        initial_pos = np.random.uniform(-1000, 1000, 3)
        
        # Minimum speed for aerial targets (considering physical limits like aircraft stall speed)
        min_flight_speed = self.max_velocity * 0.2  # Minimum flight speed is 20% of max speed
        
        if motion_type == 'constant_velocity':
            # Constant velocity: cruise speed, zero acceleration
            speed = np.random.uniform(self.max_velocity * 0.4, self.max_velocity * 0.8)  # Cruise speed range
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            initial_vel = speed * direction
            initial_acc = np.zeros(3)
            
        elif motion_type == 'constant_acceleration':
            # Constant acceleration: medium initial speed, constant acceleration
            # Ensure initial speed is within reasonable flight range
            speed = np.random.uniform(min_flight_speed, self.max_velocity * 0.7)
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            initial_vel = speed * direction
            
            # Acceleration within reasonable range
            acc_magnitude = np.random.uniform(self.max_acceleration * 0.1, self.max_acceleration * 0.3)
            acc_direction = np.random.randn(3)
            acc_direction = acc_direction / np.linalg.norm(acc_direction)
            initial_acc = acc_magnitude * acc_direction
            
        elif motion_type == 'coordinated_turn':
            # Coordinated turn: mainly horizontal plane motion, higher speed
            speed = np.random.uniform(self.max_velocity * 0.5, self.max_velocity * 0.8)  # Higher speed during turns
            # Mainly horizontal plane
            horizontal_direction = np.random.randn(2)
            horizontal_direction = horizontal_direction / np.linalg.norm(horizontal_direction)
            initial_vel = np.array([
                speed * horizontal_direction[0],
                speed * horizontal_direction[1],
                np.random.uniform(-speed * 0.05, speed * 0.05)  # Z direction speed very small (mainly horizontal turn)
            ])
            # Initial acceleration determined by centripetal acceleration
            initial_acc = np.zeros(3)
            
        elif motion_type == 'maneuvering':
            # Maneuvering motion: speed within flight range, moderate acceleration
            speed = np.random.uniform(min_flight_speed, self.max_velocity * 0.6)
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            initial_vel = speed * direction
            
            # Acceleration during maneuvering
            acc_magnitude = np.random.uniform(self.max_acceleration * 0.1, self.max_acceleration * 0.2)
            acc_direction = np.random.randn(3)
            acc_direction = acc_direction / np.linalg.norm(acc_direction)
            initial_acc = acc_magnitude * acc_direction
            
        else:
            # Default case: ensure above minimum flight speed
            speed = np.random.uniform(min_flight_speed, self.max_velocity * 0.5)
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            initial_vel = speed * direction
            initial_acc = np.random.uniform(-self.max_acceleration * 0.05, self.max_acceleration * 0.05, 3)
        
        return np.concatenate([initial_pos, initial_vel, initial_acc])
    
    def generate_constant_velocity(self, 
                                 initial_state: np.ndarray, 
                                 duration: float,
                                 process_noise: float,
                                 observation_noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate constant velocity trajectory"""
        T = int(duration / self.dt)
        states = np.zeros((T, self.state_dim))
        observations = np.zeros((T, self.obs_dim))
        
        # Reset noise state for new trajectory
        self._reset_noise_state()
        
        # State transition matrix
        F = self._build_state_transition_matrix(MotionMode.CONSTANT_VELOCITY)
        
        current_state = initial_state.copy()
        
        for t in range(T):
            # State transition
            current_state = F @ current_state
            
            # Apply kinematic constraints
            current_state = self._apply_kinematic_constraints(current_state)
            
            # Add process noise (with correlation)
            process_noise_vec = self._generate_process_noise_vector(
                process_noise, MotionMode.CONSTANT_VELOCITY
            )
            current_state += process_noise_vec
            
            # Constant velocity: force acceleration to zero (remove acceleration introduced by noise)
            current_state[6:9] = 0.0
            
            # Save state
            states[t] = current_state
            
            # Generate observation (with impulse and correlation)
            obs_noise = self._generate_observation_noise(observation_noise)
            observations[t] = current_state[0:3] + obs_noise
        
        return states, observations
    
    def generate_constant_acceleration(self, 
                                     initial_state: np.ndarray,
                                     acceleration: np.ndarray,
                                     duration: float,
                                     process_noise: float,
                                     observation_noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate constant acceleration trajectory"""
        T = int(duration / self.dt)
        states = np.zeros((T, self.state_dim))
        observations = np.zeros((T, self.obs_dim))
        
        # Reset noise state for new trajectory
        self._reset_noise_state()
        
        # State transition matrix
        F = self._build_state_transition_matrix(MotionMode.CONSTANT_ACCELERATION)
        
        current_state = initial_state.copy()
        current_state[6:9] = acceleration  # Set constant acceleration
        
        for t in range(T):
            # State transition
            current_state = F @ current_state
            
            # Apply kinematic constraints
            current_state = self._apply_kinematic_constraints(current_state)
            
            # Add process noise (with correlation)
            process_noise_vec = self._generate_process_noise_vector(
                process_noise, MotionMode.CONSTANT_ACCELERATION
            )
            current_state += process_noise_vec
            
            # Constant acceleration: restore set constant acceleration
            current_state[6:9] = acceleration
            
            # Save state
            states[t] = current_state
            
            # Generate observation (with impulse and correlation)
            obs_noise = self._generate_observation_noise(observation_noise)
            observations[t] = current_state[0:3] + obs_noise
        
        return states, observations

    def generate_coordinated_turn(self,
                                initial_state: np.ndarray,
                                duration: float,
                                process_noise: float,
                                observation_noise: float,
                                turn_rate: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate coordinated turn trajectory"""
        T = int(duration / self.dt)
        states = np.zeros((T, self.state_dim))
        observations = np.zeros((T, self.obs_dim))

        # Reset noise state for new trajectory
        self._reset_noise_state()

        # If turn rate not specified, generate randomly
        if turn_rate is None:
            max_turn_rate = np.deg2rad(30)  # Maximum turn rate 30 degrees/second
            turn_rate = np.random.uniform(-max_turn_rate, max_turn_rate)

        current_state = initial_state.copy()

        for t in range(T):
            # Extract current velocity and position
            pos = current_state[0:3].copy()
            vel = current_state[3:6].copy()
            
            # Calculate speed magnitude
            speed = np.linalg.norm(vel)
            
            if speed > 1e-6:  # Avoid division by zero
                # Coordinated turn motion update
                heading = np.arctan2(vel[1], vel[0])  # Current heading angle
                new_heading = heading + turn_rate * self.dt  # Update heading angle
                
                # Maintain speed magnitude, change velocity direction
                new_vel_x = speed * np.cos(new_heading)
                new_vel_y = speed * np.sin(new_heading)
                new_vel_z = vel[2]  # Z direction velocity remains unchanged
                
                # Update position
                new_pos_x = pos[0] + new_vel_x * self.dt
                new_pos_y = pos[1] + new_vel_y * self.dt
                new_pos_z = pos[2] + new_vel_z * self.dt
                
                # Update state
                current_state[0] = new_pos_x
                current_state[1] = new_pos_y
                current_state[2] = new_pos_z
                current_state[3] = new_vel_x
                current_state[4] = new_vel_y
                current_state[5] = new_vel_z
                
                # Acceleration under coordinated turn (centripetal acceleration)
                # a = v * ω, direction points to center
                centripetal_acc_x = -speed * turn_rate * np.sin(new_heading)
                centripetal_acc_y = speed * turn_rate * np.cos(new_heading)
                centripetal_acc_z = 0.0  # Horizontal plane turn, no centripetal acceleration in Z direction
                
                current_state[6] = centripetal_acc_x
                current_state[7] = centripetal_acc_y
                current_state[8] = centripetal_acc_z
            else:
                # If speed close to zero, use simple constant velocity motion
                current_state[0:3] += vel * self.dt
                current_state[6:9] = 0.0

            # Apply kinematic constraints
            current_state = self._apply_kinematic_constraints(current_state)
            
            # Add process noise (with correlation)
            process_noise_vec = self._generate_process_noise_vector(
                process_noise, MotionMode.COORDINATED_TURN
            )
            current_state += process_noise_vec
            
            # Recalculate centripetal acceleration (ensure matches current velocity)
            vel = current_state[3:6]
            speed = np.linalg.norm(vel)
            if speed > 1e-6:
                heading = np.arctan2(vel[1], vel[0])
                current_state[6] = -speed * turn_rate * np.sin(heading)
                current_state[7] = speed * turn_rate * np.cos(heading)
                current_state[8] = 0.0

            # Save state
            states[t] = current_state
            
            # Generate observation (with impulse and correlation)
            obs_noise = self._generate_observation_noise(observation_noise)
            observations[t] = current_state[0:3] + obs_noise

        return states, observations

    def generate_maneuvering(self,
                           initial_state: np.ndarray,
                           duration: float,
                           process_noise: float,
                           observation_noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate maneuvering trajectory (multi-mode switching among all 6 motion types)"""
        T = int(duration / self.dt)
        states = np.zeros((T, self.state_dim))
        observations = np.zeros((T, self.obs_dim))
        
        # Reset noise state for new trajectory
        self._reset_noise_state()
        
        # Transition probability matrix (6x6 for all motion types)
        # Order: CV, CA, CT, Weaving, Vertical Maneuver, Loop
        transition_matrix = np.array([
            [0.6, 0.1, 0.1, 0.05, 0.1, 0.05],  # From constant velocity
            [0.2, 0.4, 0.1, 0.1, 0.1, 0.1],    # From constant acceleration
            [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],    # From coordinated turn
            [0.1, 0.1, 0.1, 0.4, 0.1, 0.1],    # From weaving
            [0.1, 0.1, 0.05, 0.05, 0.5, 0.1],  # From vertical maneuver
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.4]     # From loop
        ])
        
        # Initial state and mode
        current_state = initial_state.copy()
        current_mode = np.random.choice(6)
        
        # Mode parameters
        current_acceleration = None
        current_turn_rate = None
        current_weaving_freq = None
        current_weaving_amp = None
        current_vertical_angle = None  # Positive for climb, negative for dive
        current_loop_radius = None
        current_loop_phase = 0.0
        current_loop_heading = None
        
        # Initialize mode parameters
        if current_mode == MotionMode.CONSTANT_ACCELERATION.value:
            acc_magnitude = np.random.uniform(self.max_acceleration * 0.1, self.max_acceleration * 0.3)
            acc_direction = np.random.randn(3)
            acc_direction = acc_direction / np.linalg.norm(acc_direction)
            current_acceleration = acc_magnitude * acc_direction
        elif current_mode == MotionMode.COORDINATED_TURN.value:
            max_turn_rate = np.deg2rad(30)
            current_turn_rate = np.random.uniform(-max_turn_rate, max_turn_rate)
        elif current_mode == MotionMode.WEAVING.value:
            current_weaving_freq = np.random.uniform(0.2, 0.5)
            current_weaving_amp = np.random.uniform(0.3, 0.6)
        elif current_mode == MotionMode.VERTICAL_MANEUVER.value:
            # Randomly choose climb (positive) or dive (negative)
            if np.random.random() < 0.5:
                current_vertical_angle = np.random.uniform(30, 80)  # Climb
            else:
                current_vertical_angle = -np.random.uniform(30, 60)  # Dive
        elif current_mode == MotionMode.LOOP.value:
            current_loop_radius = np.random.uniform(500, 1000)
            current_loop_heading = np.arctan2(current_state[4], current_state[3])
        
        # Minimum interval for mode switching
        min_mode_duration = int(2.0 / self.dt)  # Maintain at least 2 seconds
        mode_duration = 0
        
        for t in range(T):
            # Save current state
            states[t] = current_state
            
            # Generate observation (with impulse and correlation)
            obs_noise = self._generate_observation_noise(observation_noise)
            observations[t] = current_state[0:3] + obs_noise
            
            # Check if mode switch is needed
            if mode_duration >= min_mode_duration:
                # Decide whether to switch mode based on transition probability
                if np.random.random() < 0.3:  # 30% probability to consider switching
                    rand_val = np.random.random()
                    cumulative_prob = 0.0
                    
                    for next_mode in range(6):
                        cumulative_prob += transition_matrix[current_mode, next_mode]
                        if rand_val < cumulative_prob:
                            if next_mode != current_mode:
                                # Switch to new mode
                                current_mode = next_mode
                                mode_duration = 0
                                
                                # Generate parameters for new mode
                                if next_mode == MotionMode.CONSTANT_ACCELERATION.value:
                                    acc_magnitude = np.random.uniform(self.max_acceleration * 0.1, self.max_acceleration * 0.3)
                                    acc_direction = np.random.randn(3)
                                    acc_direction = acc_direction / np.linalg.norm(acc_direction)
                                    current_acceleration = acc_magnitude * acc_direction
                                elif next_mode == MotionMode.COORDINATED_TURN.value:
                                    max_turn_rate = np.deg2rad(30)
                                    current_turn_rate = np.random.uniform(-max_turn_rate, max_turn_rate)
                                elif next_mode == MotionMode.WEAVING.value:
                                    current_weaving_freq = np.random.uniform(0.2, 0.5)
                                    current_weaving_amp = np.random.uniform(0.3, 0.6)
                                elif next_mode == MotionMode.VERTICAL_MANEUVER.value:
                                    # Randomly choose climb (positive) or dive (negative)
                                    if np.random.random() < 0.5:
                                        current_vertical_angle = np.random.uniform(30, 80)  # Climb
                                    else:
                                        current_vertical_angle = -np.random.uniform(30, 60)  # Dive
                                elif next_mode == MotionMode.LOOP.value:
                                    current_loop_radius = np.random.uniform(500, 1000)
                                    current_loop_heading = np.arctan2(current_state[4], current_state[3])
                                    current_loop_phase = 0.0
                            break
                mode_duration += 1
            else:
                mode_duration += 1
            
            # Update state based on current mode
            if current_mode == MotionMode.CONSTANT_VELOCITY.value:
                # Constant velocity
                F = self._build_state_transition_matrix(MotionMode.CONSTANT_VELOCITY)
                
                current_state = F @ current_state
                current_state = self._apply_kinematic_constraints(current_state)
                
                # Add process noise (with correlation)
                process_noise_vec = self._generate_process_noise_vector(
                    process_noise, MotionMode.CONSTANT_VELOCITY
                )
                current_state += process_noise_vec
                current_state[6:9] = 0.0  # Force acceleration to zero
                
            elif current_mode == MotionMode.CONSTANT_ACCELERATION.value:
                # Constant acceleration
                F = self._build_state_transition_matrix(MotionMode.CONSTANT_ACCELERATION)
                
                current_state = F @ current_state
                current_state = self._apply_kinematic_constraints(current_state)
                
                # Add process noise (with correlation)
                process_noise_vec = self._generate_process_noise_vector(
                    process_noise, MotionMode.CONSTANT_ACCELERATION
                )
                current_state += process_noise_vec
                current_state[6:9] = current_acceleration  # Restore constant acceleration
                
            elif current_mode == MotionMode.COORDINATED_TURN.value:
                # Coordinated turn
                pos = current_state[0:3].copy()
                vel = current_state[3:6].copy()
                speed = np.linalg.norm(vel)
                
                if speed > 1e-6:
                    heading = np.arctan2(vel[1], vel[0])
                    new_heading = heading + current_turn_rate * self.dt
                    
                    new_vel_x = speed * np.cos(new_heading)
                    new_vel_y = speed * np.sin(new_heading)
                    new_vel_z = vel[2]
                    
                    new_pos_x = pos[0] + new_vel_x * self.dt
                    new_pos_y = pos[1] + new_vel_y * self.dt
                    new_pos_z = pos[2] + new_vel_z * self.dt
                    
                    current_state[0] = new_pos_x
                    current_state[1] = new_pos_y
                    current_state[2] = new_pos_z
                    current_state[3] = new_vel_x
                    current_state[4] = new_vel_y
                    current_state[5] = new_vel_z
                    
                    current_state[6] = -speed * current_turn_rate * np.sin(new_heading)
                    current_state[7] = speed * current_turn_rate * np.cos(new_heading)
                    current_state[8] = 0.0
                else:
                    current_state[0:3] += vel * self.dt
                    current_state[6:9] = 0.0
                
                current_state = self._apply_kinematic_constraints(current_state)
                
                # Add process noise (with correlation)
                process_noise_vec = self._generate_process_noise_vector(
                    process_noise, MotionMode.COORDINATED_TURN
                )
                current_state += process_noise_vec
                
                # Recalculate centripetal acceleration
                vel = current_state[3:6]
                speed = np.linalg.norm(vel)
                if speed > 1e-6:
                    heading = np.arctan2(vel[1], vel[0])
                    current_state[6] = -speed * current_turn_rate * np.sin(heading)
                    current_state[7] = speed * current_turn_rate * np.cos(heading)
                    current_state[8] = 0.0
                    
            elif current_mode == MotionMode.WEAVING.value:
                # Weaving (sinusoidal turn)
                pos = current_state[0:3].copy()
                vel = current_state[3:6].copy()
                speed = np.linalg.norm(vel)
                
                if speed > 1e-6:
                    current_time = t * self.dt
                    turn_rate = current_weaving_amp * np.sin(2 * np.pi * current_weaving_freq * current_time)
                    
                    heading = np.arctan2(vel[1], vel[0])
                    new_heading = heading + turn_rate * self.dt
                    
                    new_vel_x = speed * np.cos(new_heading)
                    new_vel_y = speed * np.sin(new_heading)
                    new_vel_z = vel[2]
                    
                    new_pos_x = pos[0] + new_vel_x * self.dt
                    new_pos_y = pos[1] + new_vel_y * self.dt
                    new_pos_z = pos[2] + new_vel_z * self.dt
                    
                    current_state[0] = new_pos_x
                    current_state[1] = new_pos_y
                    current_state[2] = new_pos_z
                    current_state[3] = new_vel_x
                    current_state[4] = new_vel_y
                    current_state[5] = new_vel_z
                    
                    current_state[6] = -speed * turn_rate * np.sin(new_heading)
                    current_state[7] = speed * turn_rate * np.cos(new_heading)
                    current_state[8] = 0.0
                else:
                    current_state[0:3] += vel * self.dt
                    current_state[6:9] = 0.0
                
                current_state = self._apply_kinematic_constraints(current_state)
                
                process_noise_vec = self._generate_process_noise_vector(
                    process_noise, MotionMode.WEAVING
                )
                current_state += process_noise_vec
                
            elif current_mode == MotionMode.VERTICAL_MANEUVER.value:
                # Vertical maneuver (climb or dive)
                vel = current_state[3:6].copy()
                speed = np.linalg.norm(vel)
                g = 9.8
                
                if speed > 1e-6:
                    angle_rad = np.deg2rad(abs(current_vertical_angle))
                    horizontal_speed = speed * np.cos(angle_rad)
                    vertical_speed = speed * np.sin(angle_rad)
                    
                    # Positive angle for climb, negative for dive
                    if current_vertical_angle > 0:
                        vertical_speed = vertical_speed  # Climb
                        gravity_compensation = g
                    else:
                        vertical_speed = -vertical_speed  # Dive
                        gravity_compensation = -g
                    
                    horizontal_dir = vel[0:2] / np.linalg.norm(vel[0:2])
                    
                    desired_vel = np.array([
                        horizontal_speed * horizontal_dir[0],
                        horizontal_speed * horizontal_dir[1],
                        vertical_speed
                    ])
                    
                    desired_acc = (desired_vel - vel) / self.dt
                    desired_acc[2] += gravity_compensation
                    
                    new_vel = vel + desired_acc * self.dt
                    new_pos = current_state[0:3] + new_vel * self.dt
                    
                    current_state[0:3] = new_pos
                    current_state[3:6] = new_vel
                    current_state[6:9] = desired_acc
                else:
                    current_state[0:3] += vel * self.dt
                    current_state[6:9] = np.array([0, 0, g])
                
                current_state = self._apply_kinematic_constraints(current_state)
                
                process_noise_vec = self._generate_process_noise_vector(
                    process_noise, MotionMode.VERTICAL_MANEUVER
                )
                current_state += process_noise_vec
                
            elif current_mode == MotionMode.LOOP.value:
                # Loop
                vel = current_state[3:6].copy()
                speed = np.linalg.norm(vel)
                
                if speed > 1e-6:
                    current_loop_phase += (speed / current_loop_radius) * self.dt
                    
                    loop_x = current_loop_radius * np.sin(current_loop_phase)
                    loop_z = current_loop_radius * (1 - np.cos(current_loop_phase))
                    
                    cos_heading = np.cos(current_loop_heading)
                    sin_heading = np.sin(current_loop_heading)
                    
                    current_state[0] += (loop_x * cos_heading) * self.dt
                    current_state[1] += (loop_x * sin_heading) * self.dt
                    current_state[2] += loop_z * self.dt
                    
                    vel_x = speed * np.cos(current_loop_phase) * cos_heading
                    vel_y = speed * np.cos(current_loop_phase) * sin_heading
                    vel_z = speed * np.sin(current_loop_phase)
                    
                    current_state[3] = vel_x
                    current_state[4] = vel_y
                    current_state[5] = vel_z
                    
                    turn_rate = speed / current_loop_radius
                    acc_x = -speed * turn_rate * np.sin(current_loop_phase) * cos_heading
                    acc_y = -speed * turn_rate * np.sin(current_loop_phase) * sin_heading
                    acc_z = speed * turn_rate * np.cos(current_loop_phase)
                    
                    current_state[6] = acc_x
                    current_state[7] = acc_y
                    current_state[8] = acc_z
                else:
                    current_state[0:3] += vel * self.dt
                    current_state[6:9] = 0.0
                
                current_state = self._apply_kinematic_constraints(current_state)
                
                process_noise_vec = self._generate_process_noise_vector(
                    process_noise, MotionMode.LOOP
                )
                current_state += process_noise_vec

        return states, observations

    def generate_vertical_maneuver(self,
                                  initial_state: np.ndarray,
                                  duration: float,
                                  process_noise: float,
                                  observation_noise: float,
                                  angle: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate vertical maneuver trajectory (climb or dive)
        
        Args:
            initial_state: Initial state [x,y,z,vx,vy,vz,ax,ay,az]
            duration: Duration of trajectory (seconds)
            process_noise: Process noise standard deviation
            observation_noise: Observation noise standard deviation
            angle: Vertical angle in degrees. Positive for climb (30-80), negative for dive (-60 to -30).
                   If not specified, randomly chooses between climb and dive.
        
        Returns:
            states: True states [T, 9]
            observations: Noisy observations [T, 3]
        """
        T = int(duration / self.dt)
        states = np.zeros((T, self.state_dim))
        observations = np.zeros((T, self.obs_dim))
        
        # Reset noise state for new trajectory
        self._reset_noise_state()
        
        # Generate random angle if not specified
        if angle is None:
            # Randomly choose climb (positive) or dive (negative)
            if np.random.random() < 0.5:
                angle = np.random.uniform(30, 80)  # Climb
            else:
                angle = -np.random.uniform(30, 60)  # Dive
        
        angle_rad = np.deg2rad(abs(angle))
        g = 9.8  # Gravity acceleration
        
        # Maximum allowed acceleration for gradual turn
        max_turn_accel = self.max_acceleration * 0.8
        
        current_state = initial_state.copy()
        
        for t in range(T):
            # Extract current velocity
            vel = current_state[3:6].copy()
            speed = np.linalg.norm(vel)
            
            if speed > 1e-6:
                # Calculate desired vertical direction
                horizontal_speed = speed * np.cos(angle_rad)
                vertical_speed = speed * np.sin(angle_rad)
                
                # Positive angle for climb, negative for dive
                if angle > 0:
                    vertical_speed = vertical_speed  # Climb
                    gravity_compensation = g
                else:
                    vertical_speed = -vertical_speed  # Dive
                    gravity_compensation = -g
                
                # Current horizontal direction
                horizontal_dir = vel[0:2] / np.linalg.norm(vel[0:2])
                
                # Desired velocity
                desired_vel = np.array([
                    horizontal_speed * horizontal_dir[0],
                    horizontal_speed * horizontal_dir[1],
                    vertical_speed
                ])
                
                # Calculate required acceleration to achieve desired velocity
                required_acc = (desired_vel - vel) / self.dt
                
                # Limit acceleration to reasonable value (gradual turn)
                required_acc_norm = np.linalg.norm(required_acc)
                if required_acc_norm > max_turn_accel:
                    required_acc = required_acc * max_turn_accel / required_acc_norm
                
                # Add gravity compensation
                desired_acc = required_acc.copy()
                desired_acc[2] += gravity_compensation
                
                # Update velocity
                new_vel = vel + desired_acc * self.dt
                
                # Update position
                new_pos = current_state[0:3] + new_vel * self.dt
                
                # Update state
                current_state[0:3] = new_pos
                current_state[3:6] = new_vel
                current_state[6:9] = desired_acc
            else:
                current_state[0:3] += vel * self.dt
                current_state[6:9] = np.array([0, 0, g])
            
            # Apply kinematic constraints
            current_state = self._apply_kinematic_constraints(current_state)
            
            # Add process noise (with correlation)
            process_noise_vec = self._generate_process_noise_vector(
                process_noise, MotionMode.VERTICAL_MANEUVER
            )
            current_state += process_noise_vec
            
            # Save state
            states[t] = current_state
            
            # Generate observation (with impulse and correlation)
            obs_noise = self._generate_observation_noise(observation_noise)
            observations[t] = current_state[0:3] + obs_noise
        
        return states, observations

    def generate_weaving(self,
                        initial_state: np.ndarray,
                        duration: float,
                        process_noise: float,
                        observation_noise: float,
                        frequency: float = None,
                        amplitude: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate weaving trajectory (sinusoidal turn)
        
        Args:
            initial_state: Initial state [x,y,z,vx,vy,vz,ax,ay,az]
            duration: Duration of trajectory (seconds)
            process_noise: Process noise standard deviation
            observation_noise: Observation noise standard deviation
            frequency: Weaving frequency (Hz), default 0.2-0.5 Hz
            amplitude: Maximum turn rate amplitude (rad/s), default 0.3-0.6 rad/s
        
        Returns:
            states: True states [T, 9]
            observations: Noisy observations [T, 3]
        """
        T = int(duration / self.dt)
        states = np.zeros((T, self.state_dim))
        observations = np.zeros((T, self.obs_dim))
        
        # Reset noise state for new trajectory
        self._reset_noise_state()
        
        # Generate random parameters if not specified
        if frequency is None:
            frequency = np.random.uniform(0.2, 0.5)  # 0.2-0.5 Hz
        if amplitude is None:
            amplitude = np.random.uniform(0.3, 0.6)  # 0.3-0.6 rad/s
        
        current_state = initial_state.copy()
        
        for t in range(T):
            # Extract current velocity and position
            pos = current_state[0:3].copy()
            vel = current_state[3:6].copy()
            
            # Calculate speed magnitude
            speed = np.linalg.norm(vel)
            
            if speed > 1e-6:
                # Calculate current time
                current_time = t * self.dt
                
                # Sinusoidal turn rate: ω(t) = amplitude * sin(2π * frequency * t)
                turn_rate = amplitude * np.sin(2 * np.pi * frequency * current_time)
                
                # Update heading angle
                heading = np.arctan2(vel[1], vel[0])
                new_heading = heading + turn_rate * self.dt
                
                # Maintain speed magnitude, change velocity direction
                new_vel_x = speed * np.cos(new_heading)
                new_vel_y = speed * np.sin(new_heading)
                new_vel_z = vel[2]  # Z direction velocity remains unchanged
                
                # Update position
                new_pos_x = pos[0] + new_vel_x * self.dt
                new_pos_y = pos[1] + new_vel_y * self.dt
                new_pos_z = pos[2] + new_vel_z * self.dt
                
                # Update state
                current_state[0] = new_pos_x
                current_state[1] = new_pos_y
                current_state[2] = new_pos_z
                current_state[3] = new_vel_x
                current_state[4] = new_vel_y
                current_state[5] = new_vel_z
                
                # Centripetal acceleration
                centripetal_acc_x = -speed * turn_rate * np.sin(new_heading)
                centripetal_acc_y = speed * turn_rate * np.cos(new_heading)
                centripetal_acc_z = 0.0
                
                current_state[6] = centripetal_acc_x
                current_state[7] = centripetal_acc_y
                current_state[8] = centripetal_acc_z
            else:
                current_state[0:3] += vel * self.dt
                current_state[6:9] = 0.0
            
            # Apply kinematic constraints
            current_state = self._apply_kinematic_constraints(current_state)
            
            # Add process noise (with correlation)
            process_noise_vec = self._generate_process_noise_vector(
                process_noise, MotionMode.WEAVING
            )
            current_state += process_noise_vec
            
            # Save state
            states[t] = current_state
            
            # Generate observation (with impulse and correlation)
            obs_noise = self._generate_observation_noise(observation_noise)
            observations[t] = current_state[0:3] + obs_noise
        
        return states, observations

    def generate_zoom_climb(self,
                           initial_state: np.ndarray,
                           duration: float,
                           process_noise: float,
                           observation_noise: float,
                           climb_angle: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate zoom climb trajectory (rapid climb using kinetic energy)
        
        Args:
            initial_state: Initial state [x,y,z,vx,vy,vz,ax,ay,az]
            duration: Duration of trajectory (seconds)
            process_noise: Process noise standard deviation
            observation_noise: Observation noise standard deviation
            climb_angle: Climb angle in degrees, default 60-80 degrees
        
        Returns:
            states: True states [T, 9]
            observations: Noisy observations [T, 3]
        """
        T = int(duration / self.dt)
        states = np.zeros((T, self.state_dim))
        observations = np.zeros((T, self.obs_dim))
        
        # Reset noise state for new trajectory
        self._reset_noise_state()
        
        # Generate random climb angle if not specified
        if climb_angle is None:
            climb_angle = np.random.uniform(60, 80)  # 60-80 degrees
        
        climb_angle_rad = np.deg2rad(climb_angle)
        g = 9.8  # Gravity acceleration
        
        # Maximum allowed acceleration for gradual turn
        max_turn_accel = self.max_acceleration * 0.8
        
        current_state = initial_state.copy()
        
        for t in range(T):
            # Extract current velocity
            vel = current_state[3:6].copy()
            speed = np.linalg.norm(vel)
            
            if speed > 1e-6:
                # Calculate desired climb direction
                # Maintain horizontal direction, add vertical component
                horizontal_speed = speed * np.cos(climb_angle_rad)
                vertical_speed = speed * np.sin(climb_angle_rad)
                
                # Current horizontal direction
                horizontal_dir = vel[0:2] / np.linalg.norm(vel[0:2])
                
                # Desired velocity
                desired_vel = np.array([
                    horizontal_speed * horizontal_dir[0],
                    horizontal_speed * horizontal_dir[1],
                    vertical_speed
                ])
                
                # Calculate required acceleration to achieve desired velocity
                required_acc = (desired_vel - vel) / self.dt
                
                # Limit acceleration to reasonable value (gradual turn)
                required_acc_norm = np.linalg.norm(required_acc)
                if required_acc_norm > max_turn_accel:
                    required_acc = required_acc * max_turn_accel / required_acc_norm
                
                # Add gravity compensation
                desired_acc = required_acc.copy()
                desired_acc[2] += g  # Counteract gravity
                
                # Update velocity
                new_vel = vel + desired_acc * self.dt
                
                # Update position
                new_pos = current_state[0:3] + new_vel * self.dt
                
                # Update state
                current_state[0:3] = new_pos
                current_state[3:6] = new_vel
                current_state[6:9] = desired_acc
            else:
                current_state[0:3] += vel * self.dt
                current_state[6:9] = np.array([0, 0, g])
            
            # Apply kinematic constraints
            current_state = self._apply_kinematic_constraints(current_state)
            
            # Add process noise (with correlation)
            process_noise_vec = self._generate_process_noise_vector(
                process_noise, MotionMode.ZOOM_CLIMB
            )
            current_state += process_noise_vec
            
            # Save state
            states[t] = current_state
            
            # Generate observation (with impulse and correlation)
            obs_noise = self._generate_observation_noise(observation_noise)
            observations[t] = current_state[0:3] + obs_noise
        
        return states, observations

    def generate_dive_attack(self,
                            initial_state: np.ndarray,
                            duration: float,
                            process_noise: float,
                            observation_noise: float,
                            dive_angle: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dive attack trajectory (rapid dive using gravity)
        
        Args:
            initial_state: Initial state [x,y,z,vx,vy,vz,ax,ay,az]
            duration: Duration of trajectory (seconds)
            process_noise: Process noise standard deviation
            observation_noise: Observation noise standard deviation
            dive_angle: Dive angle in degrees, default 30-60 degrees
        
        Returns:
            states: True states [T, 9]
            observations: Noisy observations [T, 3]
        """
        T = int(duration / self.dt)
        states = np.zeros((T, self.state_dim))
        observations = np.zeros((T, self.obs_dim))
        
        # Reset noise state for new trajectory
        self._reset_noise_state()
        
        # Generate random dive angle if not specified
        if dive_angle is None:
            dive_angle = np.random.uniform(30, 60)  # 30-60 degrees
        
        dive_angle_rad = np.deg2rad(dive_angle)
        g = 9.8  # Gravity acceleration
        
        # Maximum allowed acceleration for gradual turn
        max_turn_accel = self.max_acceleration * 0.8
        
        current_state = initial_state.copy()
        
        for t in range(T):
            # Extract current velocity
            vel = current_state[3:6].copy()
            speed = np.linalg.norm(vel)
            
            if speed > 1e-6:
                # Calculate desired dive direction
                # Maintain horizontal direction, add vertical component (downward)
                horizontal_speed = speed * np.cos(dive_angle_rad)
                vertical_speed = -speed * np.sin(dive_angle_rad)  # Negative for dive
                
                # Current horizontal direction
                horizontal_dir = vel[0:2] / np.linalg.norm(vel[0:2])
                
                # Desired velocity
                desired_vel = np.array([
                    horizontal_speed * horizontal_dir[0],
                    horizontal_speed * horizontal_dir[1],
                    vertical_speed
                ])
                
                # Calculate required acceleration to achieve desired velocity
                required_acc = (desired_vel - vel) / self.dt
                
                # Limit acceleration to reasonable value (gradual turn)
                required_acc_norm = np.linalg.norm(required_acc)
                if required_acc_norm > max_turn_accel:
                    required_acc = required_acc * max_turn_accel / required_acc_norm
                
                # Add gravity effect (gravity assists dive)
                desired_acc = required_acc.copy()
                desired_acc[2] -= g  # Gravity assists dive
                
                # Update velocity
                new_vel = vel + desired_acc * self.dt
                
                # Update position
                new_pos = current_state[0:3] + new_vel * self.dt
                
                # Update state
                current_state[0:3] = new_pos
                current_state[3:6] = new_vel
                current_state[6:9] = desired_acc
            else:
                current_state[0:3] += vel * self.dt
                current_state[6:9] = np.array([0, 0, -g])
            
            # Apply kinematic constraints
            current_state = self._apply_kinematic_constraints(current_state)
            
            # Add process noise (with correlation)
            process_noise_vec = self._generate_process_noise_vector(
                process_noise, MotionMode.DIVE_ATTACK
            )
            current_state += process_noise_vec
            
            # Save state
            states[t] = current_state
            
            # Generate observation (with impulse and correlation)
            obs_noise = self._generate_observation_noise(observation_noise)
            observations[t] = current_state[0:3] + obs_noise
        
        return states, observations

    def generate_loop(self,
                     initial_state: np.ndarray,
                     duration: float,
                     process_noise: float,
                     observation_noise: float,
                     loop_radius: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate loop trajectory (360-degree vertical turn)
        
        Args:
            initial_state: Initial state [x,y,z,vx,vy,vz,ax,ay,az]
            duration: Duration of trajectory (seconds)
            process_noise: Process noise standard deviation
            observation_noise: Observation noise standard deviation
            loop_radius: Loop radius in meters, default 500-1000 meters
        
        Returns:
            states: True states [T, 9]
            observations: Noisy observations [T, 3]
        """
        T = int(duration / self.dt)
        states = np.zeros((T, self.state_dim))
        observations = np.zeros((T, self.obs_dim))
        
        # Reset noise state for new trajectory
        self._reset_noise_state()
        
        # Generate random loop radius if not specified
        if loop_radius is None:
            loop_radius = np.random.uniform(500, 1000)  # 500-1000 meters
        
        g = 9.8  # Gravity acceleration
        
        # Initialize state
        current_state = initial_state.copy()
        
        # Calculate initial speed and heading
        initial_speed = np.linalg.norm(current_state[3:6])
        initial_heading = np.arctan2(current_state[4], current_state[3])
        
        # Calculate required turn rate for loop: ω = v / R
        turn_rate = initial_speed / loop_radius
        
        # Loop phase angle (starts at 0, goes to 2π)
        loop_phase = 0.0
        
        # Store initial position for absolute positioning
        initial_pos = current_state[0:3].copy()
        
        for t in range(T):
            # Update loop phase
            loop_phase += turn_rate * self.dt
            
            # Calculate current speed (considering gravity effect)
            # Speed varies during loop: faster at bottom, slower at top
            # Using energy conservation: v^2 = v0^2 - 2*g*h
            # where h is height change from bottom of loop
            height_change = loop_radius * (1 - np.cos(loop_phase))
            current_speed = np.sqrt(max(initial_speed**2 - 2 * g * height_change, 1.0))
            
            # Calculate position in loop (vertical plane)
            # Loop is in the plane defined by initial heading and vertical direction
            loop_x = loop_radius * np.sin(loop_phase)
            loop_z = loop_radius * (1 - np.cos(loop_phase))
            
            # Transform to global coordinates
            # Rotate loop plane to align with initial heading
            cos_heading = np.cos(initial_heading)
            sin_heading = np.sin(initial_heading)
            
            # Absolute position (not incremental)
            current_state[0] = initial_pos[0] + loop_x * cos_heading
            current_state[1] = initial_pos[1] + loop_x * sin_heading
            current_state[2] = initial_pos[2] + loop_z
            
            # Velocity update (tangent to loop, using current speed)
            vel_x = current_speed * np.cos(loop_phase) * cos_heading
            vel_y = current_speed * np.cos(loop_phase) * sin_heading
            vel_z = current_speed * np.sin(loop_phase)
            
            current_state[3] = vel_x
            current_state[4] = vel_y
            current_state[5] = vel_z
            
            # Acceleration (centripetal acceleration + gravity)
            # Centripetal acceleration: a_c = v^2 / R, direction toward center
            centripetal_acc_mag = current_speed**2 / loop_radius
            
            # Centripetal acceleration components (pointing toward center of loop)
            # Center is at (0, R) in loop coordinates
            acc_x = -centripetal_acc_mag * np.sin(loop_phase) * cos_heading
            acc_y = -centripetal_acc_mag * np.sin(loop_phase) * sin_heading
            acc_z = centripetal_acc_mag * np.cos(loop_phase)
            
            # Add gravity effect
            acc_z -= g
            
            current_state[6] = acc_x
            current_state[7] = acc_y
            current_state[8] = acc_z
            
            # Apply kinematic constraints
            current_state = self._apply_kinematic_constraints(current_state)
            
            # Add process noise (with correlation)
            process_noise_vec = self._generate_process_noise_vector(
                process_noise, MotionMode.LOOP
            )
            current_state += process_noise_vec
            
            # Save state
            states[t] = current_state
            
            # Generate observation (with impulse and correlation)
            obs_noise = self._generate_observation_noise(observation_noise)
            observations[t] = current_state[0:3] + obs_noise
        
        return states, observations

    def generate_sample(self,
                       motion_type: str = None,
                       process_noise: float = None,
                       observation_noise: float = None,
                       duration: float = None) -> TrajectorySample:
        """Generate single trajectory sample"""
        if motion_type is None:
            motion_type = np.random.choice(self.config.motion_modes)

        if process_noise is None:
            process_noise = np.random.uniform(
                self.config.process_noise_range[0],
                self.config.process_noise_range[1]
            )

        if observation_noise is None:
            observation_noise = np.random.uniform(
                self.config.observation_noise_range[0],
                self.config.observation_noise_range[1]
            )

        if duration is None:
            duration = self.config.sequence_length * self.dt

        # Generate reasonable initial state
        initial_state = self._generate_initial_state(motion_type)
        
        # Generate trajectory based on motion type
        if motion_type == 'constant_velocity':
            states, observations = self.generate_constant_velocity(
                initial_state, duration, process_noise, observation_noise
            )
        elif motion_type == 'constant_acceleration':
            # Extract acceleration from initial state
            acceleration = initial_state[6:9]
            states, observations = self.generate_constant_acceleration(
                initial_state, acceleration, duration, process_noise, observation_noise
            )
        elif motion_type == 'maneuvering':
            states, observations = self.generate_maneuvering(
                initial_state, duration, process_noise, observation_noise
            )
        elif motion_type == 'coordinated_turn':
            # Randomly generate turn rate
            turn_rate = np.random.uniform(np.deg2rad(-30), np.deg2rad(30))
            states, observations = self.generate_coordinated_turn(
                initial_state, duration, process_noise, observation_noise, turn_rate
            )
        elif motion_type == 'weaving':
            # Randomly generate weaving parameters
            frequency = np.random.uniform(0.2, 0.5)  # 0.2-0.5 Hz
            amplitude = np.random.uniform(0.3, 0.6)  # 0.3-0.6 rad/s
            states, observations = self.generate_weaving(
                initial_state, duration, process_noise, observation_noise, frequency, amplitude
            )
        elif motion_type == 'vertical_maneuver':
            # Randomly generate vertical angle (positive for climb, negative for dive)
            if np.random.random() < 0.5:
                angle = np.random.uniform(30, 80)  # Climb
            else:
                angle = -np.random.uniform(30, 60)  # Dive
            states, observations = self.generate_vertical_maneuver(
                initial_state, duration, process_noise, observation_noise, angle
            )
        elif motion_type == 'loop':
            # Randomly generate loop radius
            loop_radius = np.random.uniform(500, 1000)  # 500-1000 meters
            states, observations = self.generate_loop(
                initial_state, duration, process_noise, observation_noise, loop_radius
            )
        else:
            raise ValueError(f"Unknown motion type: {motion_type}")

        # Create sample
        sample = TrajectorySample(
            states=states,
            observations=observations,
            process_noise=process_noise,
            observation_noise=observation_noise,
            motion_type=motion_type,
            dt=self.dt
        )
        
        # Validate sample
        if not sample.validate():
            logger.warning(f"Generated sample failed validation for motion type: {motion_type}")
        
        return sample

    def generate_dataset(self,
                        num_samples: int,
                        split: str = 'train') -> List[TrajectorySample]:
        """Generate dataset"""
        dataset = []

        logger.info(f"Generating {split} dataset with {num_samples} samples...")

        for i in range(num_samples):
            if i % 100 == 0:
                logger.info(f"  Progress: {i}/{num_samples}")

            sample = self.generate_sample()
            dataset.append(sample)

        logger.info(f"Generated {len(dataset)} samples for {split} dataset")
        return dataset

    def save_dataset(self,
                    dataset: List[TrajectorySample],
                    split: str = 'train'):
        """Save dataset"""
        # Build subdirectory path using dataset_name
        base_dir = self.config.data_dir
        if hasattr(self.config, 'dataset_name') and self.config.dataset_name != "default":
            data_dir = os.path.join(base_dir, self.config.dataset_name)
        else:
            data_dir = base_dir

        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f"{split}_dataset.pkl")

        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)

        logger.info(f"Dataset saved to {file_path}")

    def load_dataset(self, split: str = 'train') -> Optional[List[TrajectorySample]]:
        """Load dataset"""
        # Build subdirectory path using dataset_name
        base_dir = self.config.data_dir
        if hasattr(self.config, 'dataset_name') and self.config.dataset_name != "default":
            data_dir = os.path.join(base_dir, self.config.dataset_name)
        else:
            data_dir = base_dir

        file_path = os.path.join(data_dir, f"{split}_dataset.pkl")

        if not os.path.exists(file_path):
            return None

        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)

        logger.info(f"Dataset loaded from {file_path}")
        return dataset

    def generate_all_datasets(self, force_regenerate: bool = False):
        """Generate all datasets (train, validation, test)"""
        # Check if already exist
        if not force_regenerate:
            train_data = self.load_dataset('train')
            val_data = self.load_dataset('val')
            test_data = self.load_dataset('test')

            if train_data is not None and val_data is not None and test_data is not None:
                logger.info("Datasets already exist. Skipping generation.")
                return train_data, val_data, test_data

        # Generate datasets
        train_data = self.generate_dataset(self.config.train_samples, 'train')
        val_data = self.generate_dataset(self.config.val_samples, 'val')
        test_data = self.generate_dataset(self.config.test_samples, 'test')

        # Save datasets
        self.save_dataset(train_data, 'train')
        self.save_dataset(val_data, 'val')
        self.save_dataset(test_data, 'test')

        return train_data, val_data, test_data

def visualize_trajectory(sample: TrajectorySample, save_path: str = None):
    """Visualize trajectory"""
    fig = plt.figure(figsize=(15, 10))

    # 3D trajectory
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(sample.states[:, 0], sample.states[:, 1], sample.states[:, 2],
             'b-', label='True trajectory', linewidth=2)
    ax1.plot(sample.observations[:, 0], sample.observations[:, 1], sample.observations[:, 2],
             'r.', label='Observations', alpha=0.6)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # Velocity curve
    ax2 = fig.add_subplot(232)
    time = np.arange(len(sample.states)) * sample.dt
    velocity = np.linalg.norm(sample.states[:, 3:6], axis=1)
    ax2.plot(time, velocity, 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Profile')
    ax2.grid(True)

    # Acceleration curve
    ax3 = fig.add_subplot(233)
    acceleration = np.linalg.norm(sample.states[:, 6:9], axis=1)
    ax3.plot(time, acceleration, 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s^2)')
    ax3.set_title('Acceleration Profile')
    ax3.grid(True)

    # X coordinate
    ax4 = fig.add_subplot(234)
    ax4.plot(time, sample.states[:, 0], 'b-', label='True', linewidth=2)
    ax4.plot(time, sample.observations[:, 0], 'r.', label='Observed', alpha=0.6)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('X (m)')
    ax4.set_title('X Position')
    ax4.legend()
    ax4.grid(True)

    # Y coordinate
    ax5 = fig.add_subplot(235)
    ax5.plot(time, sample.states[:, 1], 'b-', label='True', linewidth=2)
    ax5.plot(time, sample.observations[:, 1], 'r.', label='Observed', alpha=0.6)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('Y Position')
    ax5.legend()
    ax5.grid(True)

    # Z coordinate
    ax6 = fig.add_subplot(236)
    ax6.plot(time, sample.states[:, 2], 'b-', label='True', linewidth=2)
    ax6.plot(time, sample.observations[:, 2], 'r.', label='Observed', alpha=0.6)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Z (m)')
    ax6.set_title('Z Position')
    ax6.legend()
    ax6.grid(True)

    plt.suptitle(f'Trajectory Sample - {sample.motion_type}\n'
                 f'Process Noise: {sample.process_noise:.2f}, '
                 f'Observation Noise: {sample.observation_noise:.2f}',
                 fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Trajectory visualization saved to {save_path}")

    return fig

def visualize_multiple_trajectories(samples: List[TrajectorySample], save_path: str = None):
    """Visualize multiple trajectory samples in 3D"""
    motion_types = [
        'constant_velocity', 
        'constant_acceleration', 
        'coordinated_turn', 
        'loop',
        'weaving',
        'vertical_maneuver',
        'maneuvering'
    ]

    # Create 7x2 subplot layout with 3D projection
    fig = plt.figure(figsize=(20, 35))
    fig.suptitle('Trajectory Samples - Different Motion Types (3D View)', fontsize=16)

    for i, motion_type in enumerate(motion_types):
        # Get two samples of this motion type
        type_samples = [s for s in samples if s.motion_type == motion_type]

        for j, trajectory_sample in enumerate(type_samples[:2]):  # Take first two samples of each type
            # Create 3D subplot
            ax = fig.add_subplot(7, 2, i*2 + j + 1, projection='3d')

            # Plot 3D trajectory
            ax.plot(trajectory_sample.states[:, 0], trajectory_sample.states[:, 1], trajectory_sample.states[:, 2],
                   'b-', label='True trajectory', linewidth=2)
            ax.plot(trajectory_sample.observations[:, 0], trajectory_sample.observations[:, 1], trajectory_sample.observations[:, 2],
                   'r.', label='Observations', alpha=0.6, markersize=3)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'{motion_type.replace("_", " ").title()} - Sample {j+1}\n'
                        f'Process Noise: {trajectory_sample.process_noise:.2f}, '
                        f'Observation Noise: {trajectory_sample.observation_noise:.2f}')
            ax.legend()
            ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Multiple trajectories visualization saved to {save_path}")

    return fig

if __name__ == "__main__":
    # Test data generation
    from training.config import default_config

    generator = TrajectoryGenerator(default_config.data)

    # Generate two samples for each of the seven trajectory types
    motion_types = [
        'constant_velocity', 
        'constant_acceleration', 
        'coordinated_turn', 
        'maneuvering',
        'weaving',
        'vertical_maneuver',
        'loop'
    ]
    all_samples = []

    logger.info("Generating trajectory samples for visualization...")

    for motion_type in motion_types:
        logger.info(f"  Generating {motion_type} samples...")
        for i in range(2):  # Generate 2 samples of each type
            trajectory_sample = generator.generate_sample(motion_type=motion_type)
            all_samples.append(trajectory_sample)
            logger.info(f"    Sample {i+1}: {len(trajectory_sample.states)} time steps")

    # Visualize multiple trajectories
    os.makedirs("outputs", exist_ok=True)
    visualize_multiple_trajectories(all_samples, "all_trajectory_types.png")
    logger.info("Visualization saved to outputs/all_trajectory_types.png")
