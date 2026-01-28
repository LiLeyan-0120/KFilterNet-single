from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    """Data configuration"""
    force_regenerate: bool = False

    sequence_length: int = 250  # Sequence length
    dt: float = 0.02
    train_samples: int = 5000   # Number of training samples
    val_samples: int = 1000     # Number of validation samples
    test_samples: int = 2000    # Number of test samples

    # Dataset identifier for ablation study
    # "default", "complex_complex_noise", "complex_fixed_noise", "simple_ca_complex_noise"
    dataset_name: str = "default"

    # Motion mode parameters
    motion_modes: List[str] = None

    # Noise parameter ranges
    process_noise_range: tuple = (0.01, 0.5)    # Process noise range
    observation_noise_range: tuple = (0.1, 3.0) # Observation noise range

    # Impulse noise parameters (electronic warfare interference)
    enable_impulse_noise: bool = True           # Enable impulse noise
    impulse_probability: float = 0.05           # Probability of impulse occurrence (5%)
    impulse_magnitude_range: tuple = (5.0, 20.0) # Impulse magnitude multiplier

    # Correlated noise parameters (colored noise)
    enable_correlated_noise: bool = True        # Enable correlated noise
    noise_correlation_coeff: float = 0.4        # Correlation coefficient (0-1)

    # Kinematic constraints
    max_acceleration: float = 12    # Maximum acceleration (m/s^2)
    max_velocity: float = 500.0     # Maximum velocity (m/s)

    # Data paths
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    def __post_init__(self):
        if self.motion_modes is None:
            if self.dataset_name == "simple_ca_complex_noise":
                # Simple trajectory: includes only constant acceleration motion
                self.motion_modes = ['constant_acceleration']
            # Both complex_complex_noise and complex_fixed_noise use full mode
            else:
                # Default: full mode (CV/CA/CT/Maneuvering/Weaving/Vertical/Loop)
                self.motion_modes = [
                    'constant_velocity',
                    'constant_acceleration',
                    'coordinated_turn',
                    'maneuvering',
                    'weaving',
                    'vertical_maneuver',
                    'loop'
                ]

        # Set noise parameters based on dataset
        if self.dataset_name == "complex_fixed_noise":
            # Fixed noise: only Gaussian noise with fixed parameters
            self.enable_impulse_noise = False
            self.enable_correlated_noise = False
            # Set fixed values (using midpoint of range)
            self.process_noise_range = (0.1, 0.1)  # Fixed process noise
            self.observation_noise_range = (1.0, 1.0)  # Fixed observation noise


@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str = "KFilterNet"
    
    # Network structure parameters
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    
    # State dimensions
    state_dim: int = 9  # [x, y, z, vx, vy, vz, ax, ay, az]
    obs_dim: int = 3    # [x, y, z]

    # Module type configuration (for KFilterNet model)
    # Optional values: "learnable", "fixed", "semi_fixed"
    H_type: str = "fixed"      # Observation matrix H type
    K_type: str = "fixed"      # Kalman gain K type
    F_type: str = "fixed"      # State transition matrix F type
    Q_type: str = "learnable"  # Process noise Q type
    R_type: str = "learnable"  # Observation noise R type
    init_type: str = "fixed"   # Initial state type
    
    # Stability constraints
    stability_epsilon: float = 1e-6  # Epsilon for stability constraints


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = "adamW"  # 'adam', 'sgd', "adamW"
    scheduler: str = "cosine"  # 'cosine', 'step', 'plateau'
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Device
    device: str = "cpu"  # 'auto', 'cuda', 'cpu'
    
    # Logging and saving
    log_interval: int = 50
    save_interval: int = 50
    output_dir: str = "outputs"


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                setattr(config.data, key, value)
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                setattr(config.model, key, value)
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                setattr(config.training, key, value)
        
        return config
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        import yaml
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Default configuration instance
default_config = Config()
