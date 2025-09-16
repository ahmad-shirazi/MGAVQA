"""
Utility modules for MGA-VQA.
"""
from .metrics import compute_anls, compute_iou_metrics, MetricsTracker
from .data_loader import DocumentVQADataset, create_data_loaders, create_synthetic_data_loader
from .lr_scheduler import CosineAnnealingWarmupRestarts, create_scheduler
from .visualization import (
    visualize_attention_map, 
    visualize_bounding_boxes,
    visualize_spatial_graph,
    create_prediction_visualization,
    plot_training_curves
)

__all__ = [
    # Metrics
    'compute_anls',
    'compute_iou_metrics', 
    'MetricsTracker',
    
    # Data loading
    'DocumentVQADataset',
    'create_data_loaders',
    'create_synthetic_data_loader',
    
    # Learning rate scheduling
    'CosineAnnealingWarmupRestarts',
    'create_scheduler',
    
    # Visualization
    'visualize_attention_map',
    'visualize_bounding_boxes',
    'visualize_spatial_graph', 
    'create_prediction_visualization',
    'plot_training_curves'
]
