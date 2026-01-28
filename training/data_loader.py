import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple

from utils.data_generator import TrajectorySample

class TrajectoryDataset(Dataset):
    """Trajectory dataset"""
    
    def __init__(self, samples: List[TrajectorySample], config):
        self.samples = samples
        self.config = config
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        states = torch.FloatTensor(sample.states)  # [T, state_dim]
        observations = torch.FloatTensor(sample.observations)  # [T, obs_dim]

        return observations, states
    
    def get_data_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = False):
        """Get data loader"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

def create_data_loaders(config, force_regenerate: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders
    
    Args:
        config: Configuration object
        force_regenerate: Whether to force regenerate data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from utils.data_generator import TrajectoryGenerator
    
    # Create data generator
    generator = TrajectoryGenerator(config.data)
    
    # Generate or load data
    train_samples, val_samples, test_samples = generator.generate_all_datasets(
        force_regenerate=force_regenerate
    )
    
    # Create datasets
    train_dataset = TrajectoryDataset(train_samples, config)
    val_dataset = TrajectoryDataset(val_samples, config)
    test_dataset = TrajectoryDataset(test_samples, config)

    # Create data loaders - optimize num_workers and pin_memory
    # For GPU training, increase num_workers to speed up data loading
    device = config.training.device
    if device == 'auto':
        use_cuda = torch.cuda.is_available()
        num_workers = 4 if use_cuda else 0
        pin_memory = use_cuda
    elif device == 'cuda':
        num_workers = 4
        pin_memory = True
    else:  # device == 'cpu'
        num_workers = 0
        pin_memory = False
    
    train_loader = train_dataset.get_data_loader(
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = val_dataset.get_data_loader(
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = test_dataset.get_data_loader(
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader

class DataAugmentation:
    """Data augmentation class"""
    
    def __init__(self, config):
        self.config = config
        self.noise_scale = 0.1
    
    def add_noise(self, observations: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
        """Add random noise"""
        noise = torch.randn_like(observations) * noise_level
        return observations + noise
    
    def time_warp(self, observations: torch.Tensor, factor_range: tuple = (0.8, 1.2)) -> torch.Tensor:
        """Time warping (simple implementation)"""
        factor = np.random.uniform(factor_range[0], factor_range[1])
        new_length = int(observations.shape[1] * factor)
        
        if new_length == observations.shape[1]:
            return observations
        
        # Use linear interpolation
        result = torch.nn.functional.interpolate(
            observations.transpose(1, 2),
            size=new_length,
            mode='linear',
            align_corners=True
        ).transpose(1, 2)
        
        return result
    
    def random_crop(self, observations: torch.Tensor, crop_ratio: float = 0.9) -> torch.Tensor:
        """Random crop"""
        seq_len = observations.shape[1]
        crop_len = int(seq_len * crop_ratio)
        start_idx = np.random.randint(0, seq_len - crop_len + 1)
        
        return observations[:, start_idx:start_idx + crop_len]
    
    def apply(self, observations: torch.Tensor, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation"""
        if np.random.random() < 0.5:
            observations = self.add_noise(observations, self.noise_scale)
        
        if np.random.random() < 0.3:
            observations = self.time_warp(observations)
            # Need to adjust states length accordingly
            states = self.time_warp(states)
        
        if np.random.random() < 0.3:
            observations = self.random_crop(observations)
            states = self.random_crop(states)
        
        return observations, states
