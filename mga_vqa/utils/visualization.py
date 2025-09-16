"""
Visualization utilities for MGA-VQA.
Provides tools for visualizing attention maps, graphs, and predictions.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional, Any
import networkx as nx
from torch_geometric.data import Data
import io
import base64


def visualize_attention_map(image: np.ndarray, attention_weights: np.ndarray,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize attention weights over an image.
    
    Args:
        image: Input image [H, W, 3] 
        attention_weights: Attention weights [H', W']
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    im = axes[1].imshow(attention_weights, cmap='hot', interpolation='bilinear')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(attention_weights, cmap='hot', alpha=0.4, interpolation='bilinear')
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_bounding_boxes(image: np.ndarray, 
                           bboxes: List[List[float]], 
                           labels: List[str],
                           colors: Optional[List[str]] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize bounding boxes on an image.
    
    Args:
        image: Input image [H, W, 3]
        bboxes: List of bounding boxes [x1, y1, x2, y2] (normalized or absolute)
        labels: List of labels for each bbox
        colors: Optional list of colors for each bbox
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.imshow(image)
    
    if colors is None:
        colors = plt.cm.Set3(np.linspace(0, 1, len(bboxes)))
    
    height, width = image.shape[:2]
    
    for bbox, label, color in zip(bboxes, labels, colors):
        x1, y1, x2, y2 = bbox
        
        # Convert normalized coordinates to absolute if needed
        if all(coord <= 1.0 for coord in bbox):
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        
        # Create rectangle
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, 
                        linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        ax.text(x1, y1 - 5, label, color=color, fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    ax.set_title('Bounding Box Visualization')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_spatial_graph(graph_data: Data, pos_dict: Optional[Dict] = None,
                          node_labels: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize spatial graph structure.
    
    Args:
        graph_data: PyTorch Geometric Data object
        pos_dict: Optional position dictionary for nodes
        node_labels: Optional labels for nodes
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib figure
    """
    # Convert to NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    num_nodes = graph_data.x.shape[0]
    for i in range(num_nodes):
        G.add_node(i)
    
    # Add edges
    edge_index = graph_data.edge_index.cpu().numpy()
    edge_weights = graph_data.edge_attr.cpu().numpy().flatten() if graph_data.edge_attr is not None else None
    
    for i, (src, dst) in enumerate(edge_index.T):
        weight = edge_weights[i] if edge_weights is not None else 1.0
        G.add_edge(src, dst, weight=weight)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get positions
    if pos_dict is None:
        pos = nx.spring_layout(G, k=1, iterations=50)
    else:
        pos = pos_dict
    
    # Draw edges with varying thickness based on weights
    if edge_weights is not None:
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        # Normalize weights for visualization
        weights = np.array(weights)
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8) * 5 + 0.5
        
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color='gray')
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='gray')
    
    # Draw nodes
    node_colors = plt.cm.Set3(np.linspace(0, 1, num_nodes))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
    
    # Draw labels
    if node_labels:
        label_dict = {i: label[:10] + '...' if len(label) > 10 else label 
                     for i, label in enumerate(node_labels)}
    else:
        label_dict = {i: str(i) for i in range(num_nodes)}
    
    nx.draw_networkx_labels(G, pos, label_dict, font_size=8)
    
    ax.set_title('Spatial Graph Visualization')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_multi_modal_features(visual_features: torch.Tensor,
                                 text_features: torch.Tensor,
                                 graph_features: torch.Tensor,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize multi-modal feature distributions.
    
    Args:
        visual_features: Visual features [B, N, D]
        text_features: Text features [B, D]  
        graph_features: Graph features [B, N, D]
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert to numpy
    visual_np = visual_features.detach().cpu().numpy()
    text_np = text_features.detach().cpu().numpy()
    graph_np = graph_features.detach().cpu().numpy()
    
    # Visual features analysis
    visual_mean = visual_np.mean(axis=(0, 1))
    axes[0, 0].hist(visual_mean, bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Visual Features Distribution')
    axes[0, 0].set_xlabel('Feature Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Text features analysis
    text_mean = text_np.mean(axis=0)
    axes[0, 1].hist(text_mean, bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Text Features Distribution')
    axes[0, 1].set_xlabel('Feature Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Graph features analysis
    graph_mean = graph_np.mean(axis=(0, 1))
    axes[0, 2].hist(graph_mean, bins=50, alpha=0.7, color='red')
    axes[0, 2].set_title('Graph Features Distribution')
    axes[0, 2].set_xlabel('Feature Value')
    axes[0, 2].set_ylabel('Frequency')
    
    # Feature correlations
    # Visual feature correlation
    visual_corr = np.corrcoef(visual_np.reshape(-1, visual_np.shape[-1]).T)
    im1 = axes[1, 0].imshow(visual_corr[:50, :50], cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 0].set_title('Visual Feature Correlations')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Text-Visual similarity
    if text_np.shape[-1] == visual_np.shape[-1]:
        text_visual_sim = np.dot(text_mean, visual_mean.T) / (
            np.linalg.norm(text_mean) * np.linalg.norm(visual_mean) + 1e-8
        )
        axes[1, 1].bar(['Text-Visual'], [text_visual_sim])
        axes[1, 1].set_title('Cross-Modal Similarity')
        axes[1, 1].set_ylabel('Cosine Similarity')
    else:
        axes[1, 1].text(0.5, 0.5, 'Dimension Mismatch', ha='center', va='center')
        axes[1, 1].set_title('Cross-Modal Similarity')
    
    # Feature magnitudes
    visual_mag = np.linalg.norm(visual_mean)
    text_mag = np.linalg.norm(text_mean)
    graph_mag = np.linalg.norm(graph_mean)
    
    axes[1, 2].bar(['Visual', 'Text', 'Graph'], [visual_mag, text_mag, graph_mag])
    axes[1, 2].set_title('Feature Magnitudes')
    axes[1, 2].set_ylabel('L2 Norm')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_compression_results(original_features: torch.Tensor,
                                compressed_features: torch.Tensor,
                                relevance_scores: torch.Tensor,
                                selected_indices: Optional[torch.Tensor] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize question-guided compression results.
    
    Args:
        original_features: Original visual features [B, N, D]
        compressed_features: Compressed features [B, k, D]
        relevance_scores: Token relevance scores [B, N]
        selected_indices: Selected token indices [B, k] 
        save_path: Optional path to save visualization
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Convert to numpy
    relevance_np = relevance_scores.detach().cpu().numpy()[0]  # First sample
    
    # Relevance scores distribution
    axes[0, 0].hist(relevance_np, bins=30, alpha=0.7, color='purple')
    axes[0, 0].axvline(relevance_np.mean(), color='red', linestyle='--', 
                      label=f'Mean: {relevance_np.mean():.3f}')
    axes[0, 0].set_title('Token Relevance Scores')
    axes[0, 0].set_xlabel('Relevance Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Relevance scores over token positions
    axes[0, 1].plot(relevance_np, 'b-', alpha=0.7)
    if selected_indices is not None:
        selected_np = selected_indices.detach().cpu().numpy()[0]
        axes[0, 1].scatter(selected_np, relevance_np[selected_np], 
                          color='red', s=50, label='Selected')
    axes[0, 1].set_title('Relevance Scores by Token Position')
    axes[0, 1].set_xlabel('Token Index')
    axes[0, 1].set_ylabel('Relevance Score')
    axes[0, 1].legend()
    
    # Compression statistics
    original_count = original_features.shape[1]
    compressed_count = compressed_features.shape[1]
    compression_ratio = compressed_count / original_count
    
    stats_text = f"""
    Original tokens: {original_count}
    Compressed tokens: {compressed_count}
    Compression ratio: {compression_ratio:.2%}
    Mean relevance: {relevance_np.mean():.3f}
    Max relevance: {relevance_np.max():.3f}
    Min relevance: {relevance_np.min():.3f}
    """
    
    axes[1, 0].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    axes[1, 0].set_title('Compression Statistics')
    axes[1, 0].axis('off')
    
    # Feature magnitude comparison
    orig_mags = torch.norm(original_features[0], dim=1).detach().cpu().numpy()
    comp_mags = torch.norm(compressed_features[0], dim=1).detach().cpu().numpy()
    
    axes[1, 1].hist(orig_mags, bins=20, alpha=0.7, color='blue', label='Original', density=True)
    axes[1, 1].hist(comp_mags, bins=20, alpha=0.7, color='orange', label='Compressed', density=True)
    axes[1, 1].set_title('Feature Magnitude Distributions')
    axes[1, 1].set_xlabel('Feature Magnitude')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_prediction_visualization(image: np.ndarray,
                                  question: str,
                                  predicted_answer: str,
                                  true_answer: str,
                                  predicted_bbox: List[float],
                                  true_bbox: List[float],
                                  ocr_results: List[Dict],
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive visualization of model predictions.
    
    Args:
        image: Input image [H, W, 3]
        question: Input question text
        predicted_answer: Model's predicted answer
        true_answer: Ground truth answer
        predicted_bbox: Predicted bounding box [x1, y1, x2, y2]
        true_bbox: Ground truth bounding box [x1, y1, x2, y2]
        ocr_results: List of OCR results with 'text' and 'bbox'
        save_path: Optional path to save visualization
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    height, width = image.shape[:2]
    
    # Left plot: Image with OCR results
    axes[0].imshow(image)
    
    # Draw OCR bounding boxes
    for ocr in ocr_results:
        bbox = ocr['bbox']
        text = ocr['text']
        
        # Convert normalized to absolute coordinates if needed
        if all(coord <= 1.0 for coord in bbox):
            x1, y1, x2, y2 = [coord * width if i % 2 == 0 else coord * height 
                             for i, coord in enumerate(bbox)]
        else:
            x1, y1, x2, y2 = bbox
        
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                        linewidth=1, edgecolor='blue', facecolor='none', alpha=0.7)
        axes[0].add_patch(rect)
        
        # Add OCR text
        axes[0].text(x1, y1 - 2, text[:15], fontsize=8, color='blue',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    axes[0].set_title('Image with OCR Results')
    axes[0].axis('off')
    
    # Right plot: Predictions vs Ground Truth
    axes[1].imshow(image)
    
    # Draw ground truth bbox (green)
    if true_bbox:
        if all(coord <= 1.0 for coord in true_bbox):
            x1, y1, x2, y2 = [coord * width if i % 2 == 0 else coord * height 
                             for i, coord in enumerate(true_bbox)]
        else:
            x1, y1, x2, y2 = true_bbox
        
        gt_rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                           linewidth=3, edgecolor='green', facecolor='none', 
                           label='Ground Truth')
        axes[1].add_patch(gt_rect)
    
    # Draw predicted bbox (red)
    if predicted_bbox:
        if all(coord <= 1.0 for coord in predicted_bbox):
            x1, y1, x2, y2 = [coord * width if i % 2 == 0 else coord * height 
                             for i, coord in enumerate(predicted_bbox)]
        else:
            x1, y1, x2, y2 = predicted_bbox
        
        pred_rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=3, edgecolor='red', facecolor='none', 
                             label='Prediction', linestyle='--')
        axes[1].add_patch(pred_rect)
    
    axes[1].set_title('Predictions vs Ground Truth')
    axes[1].legend()
    axes[1].axis('off')
    
    # Add text information
    info_text = f"""
Question: {question}

Predicted Answer: {predicted_answer}
True Answer: {true_answer}

Match: {'✓' if predicted_answer.lower() == true_answer.lower() else '✗'}
    """
    
    fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(train_losses: List[float],
                        val_losses: List[float],
                        val_metrics: Dict[str, List[float]],
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses  
        val_metrics: Dictionary of validation metrics
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ANLS metric
    if 'anls' in val_metrics:
        axes[0, 1].plot(epochs, val_metrics['anls'], 'g-', label='ANLS')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('ANLS Score')
        axes[0, 1].set_title('Answer Accuracy (ANLS)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # IoU metric
    if 'iou' in val_metrics:
        axes[1, 0].plot(epochs, val_metrics['iou'], 'm-', label='IoU@0.5')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].set_title('Localization Accuracy (IoU)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Combined metrics
    ax = axes[1, 1]
    if 'anls' in val_metrics:
        ax.plot(epochs, val_metrics['anls'], 'g-', label='ANLS', alpha=0.7)
    if 'iou' in val_metrics:
        ax.plot(epochs, val_metrics['iou'], 'm-', label='IoU@0.5', alpha=0.7)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Combined Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
