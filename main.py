#!/usr/bin/env python3
"""
Main training script
For training deep learning-based Kalman filter network for single target trajectory tracking
"""

import argparse
import os
import sys
import json
import random
import numpy as np
import torch
from datetime import datetime

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize logger
from utils.logger import setup_logger
logger = setup_logger(__name__)

def set_random_seed(seed=42):
    """Set random seed for experiment reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

from training.config import Config
from training.trainer import Trainer
from models.KFilterNet_single import KFilterNet_single
from models.baseline.kalman_net_tsp import KalmanNetNN
from models.KFilterNet_single_stepInput import KFilterNet_single_stepInput
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Kalman Filter Network for Target Tracking')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='KFilterNet',
                       choices=['KFilterNet', 'KFilterNet_stepInput', 'kalman_net_tsp'],
                       help='Model type: KFilterNet, or kalman_net_tsp(pos_loss_only)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device: auto, cuda, cpu')
    
    # Data configuration
    parser.add_argument('--dataset_name', type=str, default='complex_complex_noise',
                       choices=['complex_complex_noise', 'complex_fixed_noise', 'simple_ca_complex_noise'],
                       help='Dataset name: complex_complex_noise, complex_fixed_noise, or simple_ca_complex_noise')
    parser.add_argument('--force_regenerate', action='store_true', default=False,
                       help='Force regenerate training data')
    parser.add_argument('--train_samples', type=int, default=5000,
                       help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=1000,
                       help='Number of validation samples')
    parser.add_argument('--test_samples', type=int, default=2000,
                       help='Number of test samples')
    
    # Model configuration
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    
    # Checkpoint
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def create_config(args):
    """Create configuration from arguments"""
    config = Config()
    
    # Training configuration
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.device = args.device
    
    # Data configuration
    config.data.dataset_name = args.dataset_name
    config.data.force_regenerate = args.force_regenerate
    config.data.train_samples = args.train_samples
    config.data.val_samples = args.val_samples
    config.data.test_samples = args.test_samples
    
    # Model configuration
    config.model.model_type = args.model_type
    config.model.hidden_dim = args.hidden_dim
    config.model.num_layers = args.num_layers
    config.model.dropout = args.dropout

    # Output configuration
    if args.experiment_name is None:
        args.experiment_name = f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config.training.output_dir = os.path.join(args.output_dir, args.experiment_name)
    
    return config, args

def create_model(config):
    """Create model"""
    if config.model.model_type == 'KFilterNet':
        model = KFilterNet_single(config)
    elif config.model.model_type == 'KFilterNet_stepInput':
        model = KFilterNet_single_stepInput(config)
    elif config.model.model_type == 'kalman_net_tsp':
        model = KalmanNetNN(config)
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")
    
    return model

def main():
    """Main function"""

    # Parse arguments
    args = parse_args()
    config, args = create_config(args)

    # Set random seed
    set_random_seed(42)

    # Print configuration
    logger.info("=" * 60)
    logger.info("Kalman Filter Network Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model Type: {config.model.model_type}")
    logger.info(f"Dataset Name: {config.data.dataset_name}")
    logger.info(f"Device: {config.training.device}")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Batch Size: {config.training.batch_size}")
    logger.info(f"Learning Rate: {config.training.learning_rate}")
    logger.info(f"Hidden Dim: {config.model.hidden_dim}")
    logger.info(f"Output Dir: {config.training.output_dir}")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.training.output_dir, 'config.json')
    config.save_yaml(config_path.replace('.json', '.yaml'))
    
    # Create model
    model = create_model(config)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")
    
    # Create trainer
    trainer = Trainer(config, model, device=config.training.device)
    
    # Resume checkpoint (if specified)
    if args.resume:
        if os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)
            logger.info(f"Resumed training from {args.resume}")
        else:
            logger.warning(f"Checkpoint {args.resume} not found, starting from scratch")

    try:
        # Start training
        logger.info("Starting training...")
        train_history, val_history = trainer.train()

        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {config.training.output_dir}")

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_rmse = trainer.evaluate_test_rmse()

        logger.info(f"Test Set RMSE Results:")
        logger.info(f"  Total RMSE: {test_rmse['rmse_total']:.6f}")
        logger.info(f"  X-RMSE: {test_rmse['rmse_x']:.6f}")
        logger.info(f"  Y-RMSE: {test_rmse['rmse_y']:.6f}")
        logger.info(f"  Z-RMSE: {test_rmse['rmse_z']:.6f}")
        logger.info(f"  Final Timestep RMSE: {test_rmse['final_rmse']:.6f}")
        
        # Save final configuration
        final_config = {
            'model_type': config.model.model_type,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'final_train_loss': train_history['total_loss'][-1] if train_history['total_loss'] else None,
            'final_val_loss': val_history['total_loss'][-1] if val_history['total_loss'] else None,
            'best_val_loss': trainer.best_val_loss,
            'epochs_completed': trainer.current_epoch + 1,
            'test_rmse': test_rmse
        }
        
        with open(os.path.join(config.training.output_dir, 'final_results.json'), 'w') as f:
            json.dump(final_config, f, indent=2)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save interrupted checkpoint
        trainer.save_checkpoint(is_best=False)
        logger.info(f"Interrupted checkpoint saved to {config.training.output_dir}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Save error checkpoint
        trainer.save_checkpoint(is_best=False)
        logger.info(f"Error checkpoint saved to {config.training.output_dir}")
        raise

if __name__ == "__main__":
    # Execute main function if run directly
    main()
