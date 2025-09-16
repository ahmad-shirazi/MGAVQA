"""
Question-Guided Compression for MGA-VQA.
Implements adaptive token pruning based on question relevance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class QuestionGuidedCompressor(nn.Module):
    """
    Question-Guided Compression module that prunes irrelevant visual tokens 
    based on their relevance to the input question.
    
    Implements:
    - Token relevance scoring (Equation 5)
    - Adaptive top-k selection (Equation 6)
    - Dynamic compression ratio adjustment
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.compression_ratio_min = config.compression_ratio_min
        self.compression_ratio_max = config.compression_ratio_max
        self.similarity_threshold = config.similarity_threshold
        self.importance_weight = config.importance_weight
        self.adaptive_k = config.adaptive_k
        
        # Assume visual features and question features have the same dimension
        self.feature_dim = 768  # This should match the hidden dimensions
        
        # Question-visual alignment module
        self.question_visual_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Token importance estimator
        self.importance_estimator = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Relevance scorer
        self.relevance_scorer = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Adaptive compression controller
        self.compression_controller = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4),
            nn.GELU(),
            nn.Linear(self.feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Token selection layers
        self.selection_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid()
        )
    
    def compute_similarity_scores(self, visual_tokens: torch.Tensor, 
                                 question_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores between visual tokens and question.
        
        Args:
            visual_tokens: Visual tokens [B, N, D]
            question_emb: Question embedding [B, D]
            
        Returns:
            similarity_scores: [B, N] similarity scores
        """
        if question_emb.dim() == 2 and visual_tokens.dim() == 3:
            # Expand question embedding to match visual tokens
            question_expanded = question_emb.unsqueeze(1).expand(-1, visual_tokens.shape[1], -1)
        else:
            question_expanded = question_emb
        
        # Compute cosine similarity
        similarity_scores = F.cosine_similarity(
            visual_tokens, question_expanded, dim=-1
        )  # [B, N]
        
        return similarity_scores
    
    def compute_importance_scores(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for visual tokens based on their inherent features.
        
        Args:
            visual_tokens: Visual tokens [B, N, D]
            
        Returns:
            importance_scores: [B, N] importance scores
        """
        importance_scores = self.importance_estimator(visual_tokens).squeeze(-1)  # [B, N]
        return importance_scores
    
    def compute_relevance_scores(self, visual_tokens: torch.Tensor,
                               question_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute relevance scores combining similarity and importance (Equation 5).
        
        Args:
            visual_tokens: Visual tokens [B, N, D]  
            question_emb: Question embedding [B, D]
            
        Returns:
            relevance_scores: [B, N] relevance scores
        """
        batch_size, num_tokens, feature_dim = visual_tokens.shape
        
        # Expand question embedding
        question_expanded = question_emb.unsqueeze(1).expand(-1, num_tokens, -1)
        
        # Concatenate visual tokens and question
        combined_features = torch.cat([visual_tokens, question_expanded], dim=-1)  # [B, N, 2*D]
        
        # Compute relevance scores using MLP
        relevance_scores = self.relevance_scorer(combined_features).squeeze(-1)  # [B, N]
        
        # Alternative: Combine similarity and importance explicitly
        similarity_scores = self.compute_similarity_scores(visual_tokens, question_emb)
        importance_scores = self.compute_importance_scores(visual_tokens)
        
        # Weighted combination
        combined_scores = (
            (1 - self.importance_weight) * similarity_scores + 
            self.importance_weight * importance_scores
        )
        
        # Use the maximum of MLP score and combined score
        final_scores = torch.max(relevance_scores, combined_scores)
        
        return final_scores
    
    def determine_compression_ratio(self, visual_tokens: torch.Tensor,
                                  question_emb: torch.Tensor) -> float:
        """
        Determine adaptive compression ratio based on question complexity and visual content.
        
        Args:
            visual_tokens: Visual tokens [B, N, D]
            question_emb: Question embedding [B, D]
            
        Returns:
            compression_ratio: Ratio of tokens to keep
        """
        if not self.adaptive_k:
            # Use fixed compression ratio (middle of the range)
            return (self.compression_ratio_min + self.compression_ratio_max) / 2
        
        # Global features for complexity assessment
        visual_global = visual_tokens.mean(dim=1)  # [B, D]
        
        # Combine question and visual features
        combined_global = (visual_global + question_emb) / 2  # [B, D]
        
        # Predict compression ratio
        compression_score = self.compression_controller(combined_global).squeeze(-1)  # [B]
        
        # Map to compression ratio range
        compression_ratio = (
            self.compression_ratio_min + 
            compression_score * (self.compression_ratio_max - self.compression_ratio_min)
        )
        
        # Take mean across batch for simplicity
        return compression_ratio.mean().item()
    
    def adaptive_top_k_selection(self, relevance_scores: torch.Tensor,
                               k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k tokens based on relevance scores (Equation 6).
        
        Args:
            relevance_scores: [B, N] relevance scores
            k: Number of tokens to select
            
        Returns:
            selected_indices: [B, k] indices of selected tokens
            selected_scores: [B, k] scores of selected tokens
        """
        # Get top-k indices and scores
        selected_scores, selected_indices = torch.topk(
            relevance_scores, k=k, dim=-1, largest=True, sorted=True
        )
        
        return selected_indices, selected_scores
    
    def apply_soft_attention_compression(self, visual_tokens: torch.Tensor,
                                       relevance_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply soft attention-based compression as an alternative to hard selection.
        
        Args:
            visual_tokens: Visual tokens [B, N, D]
            relevance_scores: [B, N] relevance scores
            
        Returns:
            compressed_tokens: Soft-compressed visual tokens [B, N, D]
        """
        # Normalize relevance scores to attention weights
        attention_weights = F.softmax(relevance_scores, dim=-1).unsqueeze(-1)  # [B, N, 1]
        
        # Apply attention weighting
        compressed_tokens = visual_tokens * attention_weights
        
        return compressed_tokens
    
    def forward(self, visual_tokens: torch.Tensor, 
                question_emb: torch.Tensor,
                use_hard_selection: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Question-Guided Compressor.
        
        Args:
            visual_tokens: Visual tokens from encoder [B, N, D]
            question_emb: Processed question embedding [B, D]
            use_hard_selection: Whether to use hard token selection or soft attention
            
        Returns:
            dict containing:
                - compressed_tokens: Compressed visual tokens
                - selected_indices: Indices of selected tokens (if hard selection)
                - relevance_scores: Token relevance scores
                - compression_info: Compression statistics
        """
        batch_size, num_tokens, feature_dim = visual_tokens.shape
        
        # Compute relevance scores
        relevance_scores = self.compute_relevance_scores(visual_tokens, question_emb)
        
        # Determine compression ratio and k
        compression_ratio = self.determine_compression_ratio(visual_tokens, question_emb)
        k = max(1, int(num_tokens * compression_ratio))
        
        if use_hard_selection:
            # Hard selection of top-k tokens
            selected_indices, selected_scores = self.adaptive_top_k_selection(
                relevance_scores, k
            )
            
            # Gather selected tokens
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
            compressed_tokens = visual_tokens[batch_indices, selected_indices]  # [B, k, D]
            
            # Apply selection projection
            compressed_tokens = self.selection_proj(compressed_tokens)
            
            return {
                'compressed_tokens': compressed_tokens,
                'selected_indices': selected_indices,
                'selected_scores': selected_scores,
                'relevance_scores': relevance_scores,
                'compression_info': {
                    'original_tokens': num_tokens,
                    'compressed_tokens': k,
                    'compression_ratio': compression_ratio,
                    'actual_ratio': k / num_tokens
                }
            }
        
        else:
            # Soft attention-based compression
            compressed_tokens = self.apply_soft_attention_compression(
                visual_tokens, relevance_scores
            )
            
            # Apply gating mechanism
            gates = self.gate(compressed_tokens)
            compressed_tokens = compressed_tokens * gates
            
            return {
                'compressed_tokens': compressed_tokens,
                'selected_indices': None,
                'selected_scores': None,
                'relevance_scores': relevance_scores,
                'compression_info': {
                    'original_tokens': num_tokens,
                    'compressed_tokens': num_tokens,  # Same number but attention-weighted
                    'compression_ratio': compression_ratio,
                    'actual_ratio': 1.0  # No actual token reduction in soft mode
                }
            }
    
    def get_compression_statistics(self, compression_results: Dict) -> Dict[str, float]:
        """
        Get detailed compression statistics.
        
        Args:
            compression_results: Results from forward pass
            
        Returns:
            stats: Dictionary of compression statistics
        """
        info = compression_results['compression_info']
        relevance_scores = compression_results['relevance_scores']
        
        stats = {
            'original_token_count': info['original_tokens'],
            'compressed_token_count': info['compressed_tokens'],
            'target_compression_ratio': info['compression_ratio'],
            'actual_compression_ratio': info['actual_ratio'],
            'mean_relevance_score': relevance_scores.mean().item(),
            'max_relevance_score': relevance_scores.max().item(),
            'min_relevance_score': relevance_scores.min().item(),
            'relevance_std': relevance_scores.std().item()
        }
        
        if compression_results['selected_scores'] is not None:
            selected_scores = compression_results['selected_scores']
            stats.update({
                'mean_selected_score': selected_scores.mean().item(),
                'min_selected_score': selected_scores.min().item()
            })
        
        return stats


def create_compressor(config):
    """Factory function to create question-guided compressor."""
    return QuestionGuidedCompressor(config)
