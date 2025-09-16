"""
Evaluation metrics for MGA-VQA.
Implements ANLS (Average Normalized Levenshtein Similarity) and IoU metrics.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Union
import editdistance
import re


def normalize_text(text: str) -> str:
    """Normalize text for evaluation."""
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def compute_anls(predictions: List[List[int]], targets: List[List[int]],
                vocabulary: Dict[int, str] = None) -> float:
    """
    Compute Average Normalized Levenshtein Similarity (ANLS).
    
    Args:
        predictions: List of predicted token sequences
        targets: List of target token sequences  
        vocabulary: Optional vocabulary mapping for converting tokens to text
        
    Returns:
        anls_score: Average ANLS score
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    if len(predictions) == 0:
        return 0.0
    
    anls_scores = []
    
    for pred_tokens, target_tokens in zip(predictions, targets):
        # Convert tokens to text if vocabulary is provided
        if vocabulary is not None:
            pred_text = ' '.join([vocabulary.get(token, '<UNK>') for token in pred_tokens])
            target_text = ' '.join([vocabulary.get(token, '<UNK>') for token in target_tokens])
        else:
            pred_text = ' '.join([str(token) for token in pred_tokens])
            target_text = ' '.join([str(token) for token in target_tokens])
        
        # Normalize text
        pred_text = normalize_text(pred_text)
        target_text = normalize_text(target_text)
        
        # Compute normalized edit distance
        if len(target_text) == 0:
            if len(pred_text) == 0:
                nls = 1.0
            else:
                nls = 0.0
        else:
            edit_dist = editdistance.eval(pred_text, target_text)
            nls = 1.0 - (edit_dist / max(len(pred_text), len(target_text)))
        
        anls_scores.append(max(0.0, nls))  # Ensure non-negative
    
    return float(np.mean(anls_scores))


def compute_exact_match_accuracy(predictions: List[List[int]], 
                               targets: List[List[int]]) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        predictions: List of predicted token sequences
        targets: List of target token sequences
        
    Returns:
        accuracy: Exact match accuracy
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = 0
    for pred, target in zip(predictions, targets):
        if pred == target:
            correct += 1
    
    return correct / len(predictions)


def compute_iou(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU (Intersection over Union) between bounding boxes.
    
    Args:
        bbox1: Bounding boxes [N, 4] in format (x1, y1, x2, y2)
        bbox2: Bounding boxes [N, 4] in format (x1, y1, x2, y2)
        
    Returns:
        iou: IoU scores [N]
    """
    # Compute intersection
    x1_inter = torch.max(bbox1[:, 0], bbox2[:, 0])
    y1_inter = torch.max(bbox1[:, 1], bbox2[:, 1])
    x2_inter = torch.min(bbox1[:, 2], bbox2[:, 2])
    y2_inter = torch.min(bbox1[:, 3], bbox2[:, 3])
    
    # Compute intersection area
    inter_width = torch.clamp(x2_inter - x1_inter, min=0)
    inter_height = torch.clamp(y2_inter - y1_inter, min=0)
    inter_area = inter_width * inter_height
    
    # Compute union area
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    union_area = area1 + area2 - inter_area
    
    # Compute IoU
    iou = inter_area / torch.clamp(union_area, min=1e-8)
    
    return iou


def compute_iou_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                       thresholds: List[float] = [0.5, 0.75, 0.9]) -> Dict[str, float]:
    """
    Compute IoU-based metrics at different thresholds.
    
    Args:
        predictions: Predicted bounding boxes [B, 4]
        targets: Target bounding boxes [B, 4]
        thresholds: IoU thresholds to evaluate
        
    Returns:
        metrics: Dictionary of IoU metrics
    """
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape")
    
    if predictions.numel() == 0:
        return {f'val_iou_{th}': 0.0 for th in thresholds}
    
    # Compute IoU scores
    iou_scores = compute_iou(predictions, targets)
    
    # Compute metrics at different thresholds
    metrics = {}
    
    for threshold in thresholds:
        # Accuracy at threshold
        correct = (iou_scores >= threshold).float().sum()
        accuracy = correct / len(iou_scores)
        metrics[f'val_iou_{threshold}'] = accuracy.item()
    
    # Mean IoU
    metrics['val_mean_iou'] = iou_scores.mean().item()
    
    # mAP computation (simplified)
    # Sort IoU scores in descending order
    sorted_ious, _ = torch.sort(iou_scores, descending=True)
    
    # Compute Average Precision approximation
    precisions = []
    for i, threshold in enumerate(thresholds):
        tp = (sorted_ious >= threshold).float().sum()
        precision = tp / len(sorted_ious) if len(sorted_ious) > 0 else 0.0
        precisions.append(precision.item())
    
    metrics['val_map'] = np.mean(precisions)
    
    return metrics


def compute_localization_accuracy(predictions: torch.Tensor, targets: torch.Tensor,
                                image_sizes: List[Tuple[int, int]],
                                threshold: float = 0.5) -> float:
    """
    Compute localization accuracy considering image sizes.
    
    Args:
        predictions: Normalized predicted bounding boxes [B, 4]
        targets: Normalized target bounding boxes [B, 4] 
        image_sizes: List of (width, height) for each image
        threshold: IoU threshold for correct localization
        
    Returns:
        accuracy: Localization accuracy
    """
    if len(predictions) != len(targets) or len(predictions) != len(image_sizes):
        raise ValueError("All inputs must have the same length")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = 0
    
    for pred_bbox, target_bbox, (width, height) in zip(predictions, targets, image_sizes):
        # Denormalize bounding boxes
        pred_denorm = pred_bbox * torch.tensor([width, height, width, height])
        target_denorm = target_bbox * torch.tensor([width, height, width, height])
        
        # Compute IoU
        iou = compute_iou(pred_denorm.unsqueeze(0), target_denorm.unsqueeze(0))
        
        if iou.item() >= threshold:
            correct += 1
    
    return correct / len(predictions)


def compute_retrieval_metrics(similarities: torch.Tensor, 
                            targets: torch.Tensor,
                            k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute retrieval metrics (Recall@K, MRR).
    
    Args:
        similarities: Similarity scores [B, N]
        targets: Target indices [B]
        k_values: Values of K for Recall@K
        
    Returns:
        metrics: Dictionary of retrieval metrics
    """
    batch_size, num_candidates = similarities.shape
    
    if len(targets) != batch_size:
        raise ValueError("Targets length must match batch size")
    
    # Get ranked indices (descending order)
    _, ranked_indices = torch.sort(similarities, dim=1, descending=True)
    
    metrics = {}
    
    # Compute Recall@K
    for k in k_values:
        recall_at_k = 0
        for i in range(batch_size):
            if targets[i] in ranked_indices[i, :k]:
                recall_at_k += 1
        
        metrics[f'recall@{k}'] = recall_at_k / batch_size
    
    # Compute Mean Reciprocal Rank (MRR)
    mrr_scores = []
    for i in range(batch_size):
        target_rank = (ranked_indices[i] == targets[i]).nonzero(as_tuple=True)[0]
        if len(target_rank) > 0:
            reciprocal_rank = 1.0 / (target_rank[0].item() + 1)
            mrr_scores.append(reciprocal_rank)
        else:
            mrr_scores.append(0.0)
    
    metrics['mrr'] = np.mean(mrr_scores)
    
    return metrics


class MetricsTracker:
    """Class to track and compute metrics over multiple batches."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.bbox_predictions = []
        self.bbox_targets = []
        self.image_sizes = []
    
    def update(self, predictions: List[List[int]], targets: List[List[int]],
               bbox_predictions: torch.Tensor = None, bbox_targets: torch.Tensor = None,
               image_sizes: List[Tuple[int, int]] = None):
        """Update metrics with batch results."""
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        
        if bbox_predictions is not None:
            self.bbox_predictions.append(bbox_predictions.cpu())
        if bbox_targets is not None:
            self.bbox_targets.append(bbox_targets.cpu())
        if image_sizes is not None:
            self.image_sizes.extend(image_sizes)
    
    def compute(self, vocabulary: Dict[int, str] = None) -> Dict[str, float]:
        """Compute all accumulated metrics."""
        metrics = {}
        
        # Text metrics
        if self.predictions and self.targets:
            metrics['anls'] = compute_anls(self.predictions, self.targets, vocabulary)
            metrics['exact_match'] = compute_exact_match_accuracy(self.predictions, self.targets)
        
        # Bounding box metrics
        if self.bbox_predictions and self.bbox_targets:
            all_bbox_preds = torch.cat(self.bbox_predictions, dim=0)
            all_bbox_targets = torch.cat(self.bbox_targets, dim=0)
            
            iou_metrics = compute_iou_metrics(all_bbox_preds, all_bbox_targets)
            metrics.update(iou_metrics)
            
            if self.image_sizes:
                loc_acc = compute_localization_accuracy(
                    all_bbox_preds, all_bbox_targets, self.image_sizes
                )
                metrics['localization_accuracy'] = loc_acc
        
        return metrics
