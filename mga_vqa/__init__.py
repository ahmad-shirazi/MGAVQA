"""
MGA-VQA: Multi-Modal Graph-Augmented Visual Question Answering

A PyTorch implementation of the MGA-VQA architecture for document visual question answering.
The system integrates token-level visual encoding, graph-based layout modeling, 
memory-augmented reasoning, and query-adaptive compression.
"""

__version__ = "1.0.0"
__author__ = "MGA-VQA Team"

# Main model and configuration
from .models.mga_vqa import MGA_VQA, GemmaAnswerHead, create_mga_vqa_model
from .models.baselines import BaselineEvaluator, BaselineModelFactory, create_baseline_evaluator
from .config import MGAVQAConfig, get_default_config

# Individual components
from .modules.visual_encoder import TokenLevelVisualEncoder, create_visual_encoder
from .modules.spatial_graph import SpatialGraphConstructor, GraphNeuralNetwork, create_spatial_modules
from .modules.question_processor import MemoryAugmentedQuestionProcessor, create_question_processor
from .modules.compressor import QuestionGuidedCompressor, create_compressor
from .modules.fusion import MultiModalSpatialFusion, create_fusion_module

# Training pipeline
from .training.trainer import MGAVQATrainer, MultiStageTrainingPipeline, create_trainer

# Utilities
from .utils import (
    # Metrics
    compute_anls, compute_iou_metrics, MetricsTracker,
    # Data loading
    DocumentVQADataset, create_data_loaders, create_synthetic_data_loader,
    # Learning rate scheduling
    CosineAnnealingWarmupRestarts, create_scheduler,
    # Visualization
    visualize_attention_map, visualize_bounding_boxes, visualize_spatial_graph,
    create_prediction_visualization, plot_training_curves
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Main model with Gemma-3-8B
    'MGA_VQA',
    'GemmaAnswerHead',
    'create_mga_vqa_model',
    'MGAVQAConfig', 
    'get_default_config',
    
    # Baseline models for comparison
    'BaselineEvaluator',
    'BaselineModelFactory',
    'create_baseline_evaluator',
    
    # Individual components
    'TokenLevelVisualEncoder',
    'create_visual_encoder',
    'SpatialGraphConstructor',
    'GraphNeuralNetwork',
    'create_spatial_modules',
    'MemoryAugmentedQuestionProcessor',
    'create_question_processor',
    'QuestionGuidedCompressor',
    'create_compressor',
    'MultiModalSpatialFusion',
    'create_fusion_module',
    
    # Training
    'MGAVQATrainer',
    'MultiStageTrainingPipeline',
    'create_trainer',
    
    # Utilities
    'compute_anls',
    'compute_iou_metrics',
    'MetricsTracker',
    'DocumentVQADataset',
    'create_data_loaders',
    'create_synthetic_data_loader',
    'CosineAnnealingWarmupRestarts',
    'create_scheduler',
    'visualize_attention_map',
    'visualize_bounding_boxes',
    'visualize_spatial_graph',
    'create_prediction_visualization',
    'plot_training_curves'
]

# Package-level configuration
import torch
import warnings

def set_default_device(device='auto'):
    """Set default device for MGA-VQA operations."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.set_default_device(device)
    return device

def configure_warnings(ignore_transformers=True):
    """Configure warning settings."""
    if ignore_transformers:
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Auto-configure on import
configure_warnings()
