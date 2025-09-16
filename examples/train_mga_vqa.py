#!/usr/bin/env python3
"""
Example script for training MGA-VQA model.
Demonstrates the complete training pipeline with multi-stage strategy.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from mga_vqa.config import get_default_config
from mga_vqa.training.trainer import MultiStageTrainingPipeline, MGAVQATrainer
from mga_vqa.utils.data_loader import create_data_loaders, create_synthetic_data_loader
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MGA-VQA Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/docvqa',
                       help='Directory containing dataset files')
    parser.add_argument('--use_synthetic', action='store_true',
                       help='Use synthetic data for testing')
    
    # Training arguments
    parser.add_argument('--stage', type=int, default=4, choices=[1, 2, 3, 4],
                       help='Training stage (1-4)')
    parser.add_argument('--multi_stage', action='store_true',
                       help='Run complete multi-stage training pipeline')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    
    # Model arguments
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='logs/',
                       help='Directory for logging')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                       help='Directory for saving checkpoints')
    parser.add_argument('--logger', type=str, default='tensorboard',
                       choices=['tensorboard', 'wandb'],
                       help='Logger type')
    
    # Hardware arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--precision', type=str, default='16-mixed',
                       help='Training precision')
    
    return parser.parse_args()


def setup_config(args):
    """Setup configuration from arguments."""
    config = get_default_config()
    
    # Update training config
    config.training.batch_size = args.batch_size
    config.training.max_epochs = args.max_epochs
    config.training.learning_rate = args.learning_rate
    
    # Update paths
    config.data_dir = args.data_dir
    config.log_dir = args.log_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.precision = args.precision
    
    return config


def create_data_loaders_wrapper(config, args):
    """Create data loaders based on arguments."""
    if args.use_synthetic:
        print("Using synthetic data for testing...")
        train_loader = create_synthetic_data_loader(
            batch_size=config.training.batch_size,
            num_samples=500,
            num_workers=args.num_workers
        )
        val_loader = create_synthetic_data_loader(
            batch_size=config.training.batch_size,
            num_samples=100,
            num_workers=args.num_workers
        )
        test_loader = create_synthetic_data_loader(
            batch_size=config.training.batch_size,
            num_samples=50,
            num_workers=args.num_workers
        )
        return train_loader, val_loader, test_loader
    
    else:
        print(f"Loading data from {config.data_dir}...")
        try:
            data_config = {
                'batch_size': config.training.batch_size,
                'image_size': config.visual_encoder.max_image_size,
                'max_answer_length': config.max_answer_length
            }
            
            return create_data_loaders(
                config.data_dir,
                data_config,
                num_workers=args.num_workers
            )
        
        except FileNotFoundError:
            print("Real dataset not found, falling back to synthetic data...")
            return create_data_loaders_wrapper(config, 
                                             argparse.Namespace(**{**vars(args), 'use_synthetic': True}))


def train_single_stage(config, args, data_loaders):
    """Train a single stage."""
    train_loader, val_loader, test_loader = data_loaders
    
    print(f"\nTraining MGA-VQA Stage {args.stage}")
    print("=" * 50)
    
    # Create trainer
    trainer = MGAVQATrainer(config, stage=args.stage)
    
    # Load from checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        trainer = MGAVQATrainer.load_from_checkpoint(args.checkpoint)
        trainer.stage = args.stage
        trainer.configure_training_stage()
    
    # Setup PyTorch Lightning trainer
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    
    # Logger
    if args.logger == 'wandb':
        logger = WandbLogger(
            project="mga-vqa",
            name=f"stage_{args.stage}",
            config=config.__dict__
        )
    else:
        logger = TensorBoardLogger(
            save_dir=config.log_dir,
            name=f"mga_vqa_stage_{args.stage}"
        )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.checkpoint_dir, f'stage_{args.stage}'),
        filename=f'mga_vqa_stage_{args.stage}_' + '{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.training.early_stopping_patience,
        verbose=True
    )
    
    # Lightning trainer
    pl_trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=config.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        val_check_interval=config.training.val_check_interval,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        enable_progress_bar=True
    )
    
    # Train
    pl_trainer.fit(trainer, train_loader, val_loader)
    
    # Test
    if test_loader:
        test_results = pl_trainer.test(trainer, test_loader)
        print("\nTest Results:")
        for result in test_results:
            for key, value in result.items():
                print(f"  {key}: {value:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    
    return trainer


def train_multi_stage(config, args, data_loaders):
    """Run complete multi-stage training pipeline."""
    train_loader, val_loader, test_loader = data_loaders
    
    print("\nStarting Multi-Stage MGA-VQA Training Pipeline")
    print("=" * 50)
    
    # Create pipeline
    pipeline = MultiStageTrainingPipeline(
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        logger_type=args.logger
    )
    
    # Run full pipeline
    final_model = pipeline.run_full_pipeline()
    
    # Evaluate final model
    pipeline.evaluate_final_model()
    
    return final_model


def main():
    """Main function."""
    args = parse_args()
    
    # Setup configuration
    config = setup_config(args)
    
    print("MGA-VQA Training Script")
    print("=" * 50)
    print(f"Training stage: {args.stage}")
    print(f"Multi-stage: {args.multi_stage}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Max epochs: {config.training.max_epochs}")
    print(f"Device: {config.device}")
    print(f"Precision: {config.precision}")
    
    # Create data loaders
    data_loaders = create_data_loaders_wrapper(config, args)
    
    # Create directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Train
    if args.multi_stage:
        model = train_multi_stage(config, args, data_loaders)
    else:
        model = train_single_stage(config, args, data_loaders)
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
