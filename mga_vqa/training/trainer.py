"""
Multi-Stage Training Pipeline for MGA-VQA.
Implements the training strategy described in the research paper.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from typing import Dict, List, Optional, Tuple, Any
import os
import sys
import wandb
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mga_vqa.models.mga_vqa import MGA_VQA
from mga_vqa.config import MGAVQAConfig
from mga_vqa.utils.metrics import compute_anls, compute_iou_metrics
from mga_vqa.utils.lr_scheduler import CosineAnnealingWarmupRestarts


class MGAVQATrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for MGA-VQA with multi-stage training strategy.
    
    Multi-stage training:
    1. Stage 1: Pretrain token encoder on document-specific datasets
    2. Stage 2: Supervise spatial graph module with explicit layout signals  
    3. Stage 3: Memory system integration on question-answering pairs
    4. Stage 4: End-to-end joint fine-tuning
    """
    
    def __init__(self, config: MGAVQAConfig, stage: int = 4):
        super().__init__()
        self.config = config
        self.stage = stage
        self.save_hyperparameters()
        
        # Initialize model
        self.model = MGA_VQA(config)
        
        # Training stage configuration
        self.configure_training_stage()
        
        # Metrics tracking
        self.train_metrics = {'loss': [], 'answer_loss': [], 'bbox_loss': []}
        self.val_metrics = {'loss': [], 'anls': [], 'iou': []}
        
        # Best metrics tracking
        self.best_val_anls = 0.0
        self.best_val_iou = 0.0
        
    def configure_training_stage(self):
        """Configure training parameters for each stage."""
        if self.stage == 1:
            # Stage 1: Pretrain visual encoder
            self.stage_name = "visual_encoder_pretraining"
            self.freeze_components(['graph_constructor', 'graph_reasoner', 
                                  'question_processor', 'compressor', 'fusion_module'])
            self.loss_weights = {'answer': 0.0, 'bbox': 0.0, 'visual': 1.0}
            
        elif self.stage == 2:
            # Stage 2: Spatial graph module training
            self.stage_name = "spatial_graph_training" 
            self.freeze_components(['visual_encoder', 'question_processor', 
                                  'compressor', 'fusion_module'])
            self.loss_weights = {'answer': 0.0, 'bbox': 1.0, 'visual': 0.0}
            
        elif self.stage == 3:
            # Stage 3: Memory system integration
            self.stage_name = "memory_integration"
            self.freeze_components(['visual_encoder', 'graph_constructor', 'graph_reasoner'])
            self.loss_weights = {'answer': 1.0, 'bbox': 0.0, 'visual': 0.0}
            
        elif self.stage == 4:
            # Stage 4: End-to-end fine-tuning
            self.stage_name = "end_to_end_finetuning"
            self.unfreeze_all_components()
            self.loss_weights = {'answer': 1.0, 'bbox': 1.0, 'visual': 0.0}
            
        else:
            raise ValueError(f"Invalid training stage: {self.stage}")
    
    def freeze_components(self, component_names: List[str]):
        """Freeze specified model components."""
        for name in component_names:
            if hasattr(self.model, name):
                component = getattr(self.model, name)
                for param in component.parameters():
                    param.requires_grad = False
    
    def unfreeze_all_components(self):
        """Unfreeze all model components."""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Different learning rates for different components
        param_groups = []
        
        if self.stage == 1:
            # Focus on visual encoder
            param_groups.append({
                'params': self.model.visual_encoder.parameters(),
                'lr': self.config.training.learning_rate
            })
        
        elif self.stage == 2:
            # Focus on graph components
            param_groups.extend([
                {
                    'params': self.model.graph_constructor.parameters(), 
                    'lr': self.config.training.learning_rate
                },
                {
                    'params': self.model.graph_reasoner.parameters(),
                    'lr': self.config.training.learning_rate
                },
                {
                    'params': self.model.bbox_head.parameters(),
                    'lr': self.config.training.learning_rate
                }
            ])
        
        elif self.stage == 3:
            # Focus on question processing and memory
            param_groups.extend([
                {
                    'params': self.model.question_processor.parameters(),
                    'lr': self.config.training.learning_rate
                },
                {
                    'params': self.model.compressor.parameters(),
                    'lr': self.config.training.learning_rate * 0.5
                },
                {
                    'params': self.model.fusion_module.parameters(),
                    'lr': self.config.training.learning_rate * 0.5
                },
                {
                    'params': self.model.answer_head.parameters(),
                    'lr': self.config.training.learning_rate
                }
            ])
        
        else:
            # End-to-end training with different rates for different components
            param_groups.extend([
                {
                    'params': self.model.visual_encoder.parameters(),
                    'lr': self.config.training.learning_rate * 0.1  # Lower LR for pretrained
                },
                {
                    'params': list(self.model.graph_constructor.parameters()) + 
                             list(self.model.graph_reasoner.parameters()),
                    'lr': self.config.training.learning_rate * 0.5
                },
                {
                    'params': list(self.model.question_processor.parameters()) +
                             list(self.model.compressor.parameters()) +
                             list(self.model.fusion_module.parameters()),
                    'lr': self.config.training.learning_rate
                },
                {
                    'params': list(self.model.answer_head.parameters()) +
                             list(self.model.bbox_head.parameters()),
                    'lr': self.config.training.learning_rate
                }
            ])
        
        # Use AdamW optimizer
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay
        )
        
        # Cosine annealing scheduler with warmup
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.config.training.max_epochs,
            cycle_mult=1.0,
            max_lr=self.config.training.learning_rate,
            min_lr=self.config.training.learning_rate * 0.01,
            warmup_steps=self.config.training.warmup_steps,
            gamma=0.5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step",
                "frequency": 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self.model(
            image=batch['image'],
            question_text=batch['question'],
            ocr_results=batch['ocr_results'],
            image_sizes=batch['image_sizes'],
            answer_labels=batch.get('answer_labels'),
            bbox_targets=batch.get('bbox_targets')
        )
        
        # Compute weighted loss based on training stage
        total_loss = 0
        loss_dict = {}
        
        if outputs['answer_loss'] is not None and self.loss_weights['answer'] > 0:
            answer_loss = outputs['answer_loss'] * self.loss_weights['answer']
            total_loss += answer_loss
            loss_dict['train_answer_loss'] = answer_loss
        
        if outputs['bbox_loss'] is not None and self.loss_weights['bbox'] > 0:
            bbox_loss = outputs['bbox_loss'] * self.loss_weights['bbox']
            total_loss += bbox_loss
            loss_dict['train_bbox_loss'] = bbox_loss
        
        # Add regularization losses if needed
        if self.stage == 1:
            # Add visual encoder specific losses
            visual_reg_loss = self.compute_visual_regularization_loss()
            total_loss += visual_reg_loss * 0.1
            loss_dict['train_visual_reg_loss'] = visual_reg_loss
        
        loss_dict['train_total_loss'] = total_loss
        
        # Log metrics
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store metrics
        self.train_metrics['loss'].append(total_loss.item())
        if 'train_answer_loss' in loss_dict:
            self.train_metrics['answer_loss'].append(loss_dict['train_answer_loss'].item())
        if 'train_bbox_loss' in loss_dict:
            self.train_metrics['bbox_loss'].append(loss_dict['train_bbox_loss'].item())
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self.model(
            image=batch['image'],
            question_text=batch['question'],
            ocr_results=batch['ocr_results'], 
            image_sizes=batch['image_sizes'],
            answer_labels=batch.get('answer_labels'),
            bbox_targets=batch.get('bbox_targets')
        )
        
        # Compute validation loss
        val_loss = 0
        if outputs['total_loss'] is not None:
            val_loss = outputs['total_loss']
        
        # Compute evaluation metrics
        metrics = {}
        
        # ANLS for answer accuracy (only if we have answer labels)
        if 'answer_labels' in batch and self.loss_weights['answer'] > 0:
            predicted_answers = self.model.generate_answer(outputs['answer_logits'])
            true_answers = batch['answer_labels']
            anls_score = compute_anls(predicted_answers, true_answers)
            metrics['val_anls'] = anls_score
        
        # IoU for bounding box accuracy (only if we have bbox targets)
        if 'bbox_targets' in batch and self.loss_weights['bbox'] > 0:
            bbox_predictions = outputs['bbox_predictions']
            bbox_targets = batch['bbox_targets']
            iou_metrics = compute_iou_metrics(bbox_predictions, bbox_targets)
            metrics.update(iou_metrics)
        
        metrics['val_loss'] = val_loss
        
        # Log metrics
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        
        return {'val_loss': val_loss, 'metrics': metrics}
    
    def validation_epoch_end(self, outputs):
        """End of validation epoch."""
        # Compute average metrics
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        # Track best metrics
        val_metrics = {}
        
        if any('val_anls' in x['metrics'] for x in outputs):
            avg_anls = np.mean([x['metrics']['val_anls'] for x in outputs if 'val_anls' in x['metrics']])
            val_metrics['val_anls'] = avg_anls
            
            if avg_anls > self.best_val_anls:
                self.best_val_anls = avg_anls
        
        if any('val_iou_0.5' in x['metrics'] for x in outputs):
            avg_iou = np.mean([x['metrics']['val_iou_0.5'] for x in outputs if 'val_iou_0.5' in x['metrics']])
            val_metrics['val_iou'] = avg_iou
            
            if avg_iou > self.best_val_iou:
                self.best_val_iou = avg_iou
        
        # Store metrics
        self.val_metrics['loss'].append(avg_val_loss.item())
        if 'val_anls' in val_metrics:
            self.val_metrics['anls'].append(val_metrics['val_anls'])
        if 'val_iou' in val_metrics:
            self.val_metrics['iou'].append(val_metrics['val_iou'])
        
        # Log best metrics
        self.log('best_val_anls', self.best_val_anls)
        self.log('best_val_iou', self.best_val_iou)
    
    def compute_visual_regularization_loss(self):
        """Compute regularization loss for visual encoder pretraining."""
        # Multi-scale consistency loss
        reg_loss = 0
        
        # Add consistency losses between different scales
        if hasattr(self.model.visual_encoder, 'multi_scale_patcher'):
            # This would require accessing intermediate features
            # Simplified version here
            reg_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
        
        return reg_loss
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        summary = {
            'stage': self.stage,
            'stage_name': self.stage_name,
            'best_val_anls': self.best_val_anls,
            'best_val_iou': self.best_val_iou,
            'train_metrics': {
                'avg_loss': np.mean(self.train_metrics['loss']) if self.train_metrics['loss'] else 0,
                'final_loss': self.train_metrics['loss'][-1] if self.train_metrics['loss'] else 0
            },
            'val_metrics': {
                'avg_loss': np.mean(self.val_metrics['loss']) if self.val_metrics['loss'] else 0,
                'final_anls': self.val_metrics['anls'][-1] if self.val_metrics['anls'] else 0,
                'final_iou': self.val_metrics['iou'][-1] if self.val_metrics['iou'] else 0
            }
        }
        return summary


class MultiStageTrainingPipeline:
    """
    Multi-stage training pipeline manager.
    Orchestrates training across all four stages.
    """
    
    def __init__(self, config: MGAVQAConfig, 
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 test_dataloader: Optional[DataLoader] = None,
                 logger_type: str = 'tensorboard'):
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.logger_type = logger_type
        
        # Training results
        self.stage_results = {}
        self.final_model = None
    
    def setup_logger(self, stage: int):
        """Setup logger for training stage."""
        if self.logger_type == 'wandb':
            logger = WandbLogger(
                project="mga-vqa",
                name=f"stage_{stage}",
                config=self.config.__dict__
            )
        else:
            logger = TensorBoardLogger(
                save_dir=self.config.log_dir,
                name=f"mga_vqa_stage_{stage}"
            )
        return logger
    
    def setup_callbacks(self, stage: int):
        """Setup callbacks for training stage."""
        callbacks = []
        
        # Model checkpointing
        if stage <= 3:
            monitor_metric = 'val_loss'
            mode = 'min'
        else:
            monitor_metric = 'val_anls'
            mode = 'max'
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.config.checkpoint_dir, f'stage_{stage}'),
            filename=f'mga_vqa_stage_{stage}_' + '{epoch:02d}-{val_loss:.2f}',
            monitor=monitor_metric,
            mode=mode,
            save_top_k=self.config.training.save_top_k,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=self.config.training.early_stopping_patience,
            mode=mode,
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        return callbacks
    
    def train_stage(self, stage: int, 
                   checkpoint_path: Optional[str] = None) -> MGAVQATrainer:
        """Train a specific stage."""
        print(f"\n=== Starting Training Stage {stage} ===")
        
        # Setup trainer
        trainer_model = MGAVQATrainer(self.config, stage=stage)
        
        # Load from checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            trainer_model = MGAVQATrainer.load_from_checkpoint(checkpoint_path)
            trainer_model.stage = stage
            trainer_model.configure_training_stage()
        
        # Setup logger and callbacks
        logger = self.setup_logger(stage)
        callbacks = self.setup_callbacks(stage)
        
        # Setup PyTorch Lightning trainer
        trainer = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision=self.config.precision,
            gradient_clip_val=self.config.training.gradient_clip_val,
            accumulate_grad_batches=self.config.training.accumulate_grad_batches,
            val_check_interval=self.config.training.val_check_interval,
            logger=logger,
            callbacks=callbacks,
            deterministic=False,
            enable_progress_bar=True
        )
        
        # Train
        trainer.fit(
            trainer_model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )
        
        # Store results
        self.stage_results[stage] = trainer_model.get_training_summary()
        
        print(f"=== Completed Training Stage {stage} ===")
        print(f"Best Val ANLS: {trainer_model.best_val_anls:.4f}")
        print(f"Best Val IoU: {trainer_model.best_val_iou:.4f}")
        
        return trainer_model
    
    def run_full_pipeline(self, stage_checkpoints: Optional[Dict[int, str]] = None):
        """Run the complete multi-stage training pipeline."""
        print("Starting Multi-Stage MGA-VQA Training Pipeline")
        print("=" * 50)
        
        if stage_checkpoints is None:
            stage_checkpoints = {}
        
        # Stage 1: Visual encoder pretraining
        stage1_model = self.train_stage(1, stage_checkpoints.get(1))
        
        # Stage 2: Spatial graph training (load from stage 1)
        stage2_checkpoint = stage_checkpoints.get(2)
        if stage2_checkpoint is None:
            # Use best checkpoint from stage 1
            stage2_checkpoint = stage1_model.trainer.checkpoint_callback.best_model_path
        stage2_model = self.train_stage(2, stage2_checkpoint)
        
        # Stage 3: Memory integration (load from stage 2)
        stage3_checkpoint = stage_checkpoints.get(3)
        if stage3_checkpoint is None:
            stage3_checkpoint = stage2_model.trainer.checkpoint_callback.best_model_path
        stage3_model = self.train_stage(3, stage3_checkpoint)
        
        # Stage 4: End-to-end fine-tuning (load from stage 3)
        stage4_checkpoint = stage_checkpoints.get(4)
        if stage4_checkpoint is None:
            stage4_checkpoint = stage3_model.trainer.checkpoint_callback.best_model_path
        final_model = self.train_stage(4, stage4_checkpoint)
        
        self.final_model = final_model
        
        # Print final results
        print("\n" + "=" * 50)
        print("Multi-Stage Training Pipeline Complete!")
        print("=" * 50)
        
        for stage, results in self.stage_results.items():
            print(f"Stage {stage} ({results['stage_name']}):")
            print(f"  Best Val ANLS: {results['best_val_anls']:.4f}")
            print(f"  Best Val IoU: {results['best_val_iou']:.4f}")
        
        return self.final_model
    
    def evaluate_final_model(self):
        """Evaluate the final model on test set."""
        if self.final_model is None or self.test_dataloader is None:
            print("No final model or test dataloader available for evaluation")
            return None
        
        print("\nEvaluating final model on test set...")
        
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=False
        )
        
        test_results = trainer.test(self.final_model, self.test_dataloader)
        
        print("Test Results:")
        for result in test_results:
            for key, value in result.items():
                print(f"  {key}: {value:.4f}")
        
        return test_results


def create_trainer(config: MGAVQAConfig, stage: int = 4) -> MGAVQATrainer:
    """Factory function to create MGA-VQA trainer."""
    return MGAVQATrainer(config, stage)
