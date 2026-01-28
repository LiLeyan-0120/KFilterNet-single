import torch
import torch.optim as optim
from tqdm import tqdm
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .data_loader import create_data_loaders
from .config import Config
from utils.visualization import visualize_trajectory_comparison
from utils.logger import setup_logger
from sklearn.metrics import mean_squared_error


class Trainer:
    """Trainer class for Kalman Filter Network"""
    
    def __init__(self, config: Config, model, device: str = 'auto'):
        self.config = config
        self.model = model
        self.logger = setup_logger(__name__)
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            config, force_regenerate=config.data.force_regenerate
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Logging and checkpoints
        self.output_dir = config.training.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training history - only track used losses
        self.train_history = {
            'total_loss': [],
            'state_loss': []
        }
        self.val_history = {
            'total_loss': [],
            'state_loss': []
        }
        
        # Early stopping
        self.early_stopping_patience = config.training.early_stopping_patience
        self.early_stopping_min_delta = config.training.early_stopping_min_delta
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def _create_optimizer(self):
        """Create optimizer"""
        if self.config.training.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        elif self.config.training.optimizer == 'adamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.training.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs
            )
        elif self.config.training.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.training.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            return None
    
    def compute_loss(self,
                    states_pred: torch.Tensor,
                    states_true: torch.Tensor,
                    covariances: torch.Tensor,
                    info: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss function - simplified version using only state MSE loss
        
        Args:
            states_pred: [B, T_pred, state_dim] Predicted states
            states_true: [B, T_true, state_dim] True states
            covariances: [B, T_pred, state_dim, state_dim] Covariance matrices
            info: Dict Intermediate information
        
        Returns:
            total_loss: Total loss
            loss_dict: Individual loss components
        """
        loss_dict = {}
        
        # Handle sequence length mismatch
        if states_pred.shape[1] != states_true.shape[1]:
            states_true = states_true[:, -states_pred.shape[1]:, :]

        # State error MSE (L2 loss) - only position (first 3 dimensions)
        state_error = states_pred[:, :, :3] - states_true[:, :, :3]
        loss_state = torch.mean(state_error ** 2)
        loss_dict['state_loss'] = loss_state.item()

        # Total loss - only use state_loss
        total_loss = loss_state
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def train_epoch(self) -> Dict:
        """Train one epoch"""
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (observations, states_true) in enumerate(progress_bar):
            # Move data to device
            observations = observations.to(self.device, non_blocking=True)
            states_true = states_true.to(self.device, non_blocking=True)
            
            # Forward pass
            states_pred, covariances, info = self.model(observations)
            
            # Compute loss
            loss, loss_dict = self.compute_loss(states_pred, states_true, covariances, info)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record losses
            epoch_losses.append(loss_dict)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss_dict["total_loss"]:.4f}',
                'State': f'{loss_dict["state_loss"]:.4f}'
            })
            
            # Record training log
            if self.global_step % self.config.training.log_interval == 0:
                self.log_training_info(loss_dict, 'train')
            
            self.global_step += 1
        
        # Compute average losses
        avg_losses = self._average_losses(epoch_losses)
        return avg_losses
    
    def validate_epoch(self) -> Dict:
        """Validate one epoch"""
        self.model.eval()
        epoch_losses = []
        
        with torch.no_grad():
            for observations, states_true in self.val_loader:
                # Move data to device
                observations = observations.to(self.device, non_blocking=True)
                states_true = states_true.to(self.device, non_blocking=True)
                
                # Forward pass
                states_pred, covariances, info = self.model(observations)
                
                # Compute loss
                loss, loss_dict = self.compute_loss(states_pred, states_true, covariances, info)
                epoch_losses.append(loss_dict)
        
        # Compute average losses
        avg_losses = self._average_losses(epoch_losses)
        return avg_losses
    
    def _average_losses(self, loss_list: list) -> Dict:
        """Average loss list"""
        if not loss_list:
            return {}
        
        avg_losses = {}
        for key in loss_list[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in loss_list])
        return avg_losses
    
    def log_training_info(self, losses: Dict, phase: str):
        """Record training information to log file"""
        log_file = os.path.join(self.output_dir, 'training_log.txt')
        
        with open(log_file, 'a') as f:
            f.write(f"Step {self.global_step} ({phase}):\n")
            for key, value in losses.items():
                f.write(f"  {key}: {value:.6f}\n")
            f.write("\n")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = os.path.join(self.output_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_checkpoint_path)
        
        # Periodic save
        if self.current_epoch % self.config.training.save_interval == 0:
            epoch_checkpoint_path = os.path.join(
                self.output_dir, 
                f'checkpoint_epoch_{self.current_epoch}.pth'
            )
            torch.save(checkpoint, epoch_checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        self.best_val_loss = checkpoint['best_val_loss']

        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """Check early stopping"""
        if val_loss < self.best_val_loss - self.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                return True
            return False
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(f"Model type: {self.config.model.model_type}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            val_losses = self.validate_epoch()
            
            # Record history
            for key in self.train_history.keys():
                if key in train_losses:
                    self.train_history[key].append(train_losses[key])
                if key in val_losses:
                    self.val_history[key].append(val_losses[key])
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total_loss'])
                else:
                    self.scheduler.step()
            
            # Print log
            self.logger.info(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            self.logger.info(f"  Train Loss: {train_losses['total_loss']:.4f}")
            self.logger.info(f"  Val Loss: {val_losses['total_loss']:.4f}")
            self.logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_losses['total_loss'] < self.best_val_loss
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping check
            if self.check_early_stopping(val_losses['total_loss']):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Load best checkpoint for evaluation
        best_checkpoint_path = os.path.join(self.output_dir, 'best_checkpoint.pth')
        if os.path.exists(best_checkpoint_path):
            self.logger.info(f"Loading best checkpoint for evaluation...")
            self.load_checkpoint(best_checkpoint_path)
            self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        else:
            self.logger.warning(f"Best checkpoint not found, using current model state")
        
        # Save training history
        self.save_training_history()
        
        # Visualize results
        self.visualize_training_results()
        
        return self.train_history, self.val_history
    
    def save_training_history(self):
        """Save training history"""
        history_path = os.path.join(self.output_dir, 'training_history.json')
        
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        self.logger.info(f"Training history saved to {history_path}")
    
    def visualize_training_results(self):
        """Visualize training results"""
        # Plot loss curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Total loss
        axes[0].plot(self.train_history['total_loss'], label='Train', alpha=0.8)
        axes[0].plot(self.val_history['total_loss'], label='Val', alpha=0.8)
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # State loss
        axes[1].plot(self.train_history['state_loss'], label='Train', alpha=0.8)
        axes[1].plot(self.val_history['state_loss'], label='Val', alpha=0.8)
        axes[1].set_title('State Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize trajectory comparison
        self.visualize_trajectories()
    
    def visualize_trajectories(self):
        """Visualize trajectory comparison for test samples"""
        self.model.eval()
        
        # Get test data
        test_dataset = self.test_loader.dataset
        
        # Randomly select some samples for visualization
        num_samples = min(5, len(test_dataset))
        sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        results = []
        
        with torch.no_grad():
            for idx in sample_indices:
                observations, states_true = test_dataset[idx]
                observations = observations.unsqueeze(0).to(self.device)
                
                # Model prediction
                states_pred, covariances, info = self.model(observations)

                results.append({
                    'observations': observations.cpu().squeeze(0).numpy(),
                    'states_pred': states_pred.cpu().squeeze(0).numpy(),
                    'states_true': states_true.numpy(),
                    'covariances': covariances.cpu().squeeze(0).numpy(),
                    'info': info
                })
        
        # Visualize trajectory comparison
        visualize_trajectory_comparison(
            results,
            save_path=os.path.join(self.output_dir, 'trajectory_comparison.png')
        )

        self.logger.info(f"Trajectory visualization saved to {self.output_dir}")
    
    def evaluate_test_rmse(self):
        """Evaluate RMSE on test set"""
        self.model.eval()
        test_dataset = self.test_loader.dataset

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for observations, states_true in self.test_loader:
                observations = observations.to(self.device)
                states_true = states_true.to(self.device)

                # Model prediction
                states_pred, covariances, info = self.model(observations)

                # Handle sequence length mismatch
                if states_pred.shape[1] != states_true.shape[1]:
                    states_true = states_true[:, -states_pred.shape[1]:, :]

                # Collect position predictions (first 3 dimensions)
                all_predictions.append(states_pred[:, :, :3].cpu().numpy())
                all_targets.append(states_true[:, :, :3].cpu().numpy())

        # Merge all batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate RMSE
        pred_flat = all_predictions.reshape(-1, 3)
        target_flat = all_targets.reshape(-1, 3)

        # RMSE for each dimension
        rmse_x = np.sqrt(mean_squared_error(target_flat[:, 0], pred_flat[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(target_flat[:, 1], pred_flat[:, 1]))
        rmse_z = np.sqrt(mean_squared_error(target_flat[:, 2], pred_flat[:, 2]))

        # Total RMSE
        rmse_total = np.sqrt(mean_squared_error(target_flat, pred_flat))

        # RMSE per timestep
        rmse_per_timestep = []
        for t in range(all_predictions.shape[1]):
            rmse_t = np.sqrt(mean_squared_error(
                all_targets[:, t, :],
                all_predictions[:, t, :]
            ))
            rmse_per_timestep.append(rmse_t)

        return {
            'rmse_total': rmse_total,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'rmse_z': rmse_z,
            'rmse_per_timestep': rmse_per_timestep,
            'final_rmse': rmse_per_timestep[-1] if rmse_per_timestep else rmse_total
        }
