"""
Multi-Modal Spatial Fusion for MGA-VQA.
Implements disentangled attention mechanisms for multi-modal integration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class DisentangledAttention(nn.Module):
    """
    Disentangled Attention mechanism that separates different types of interactions:
    - Text-to-Text: Intra-linguistic dependencies
    - Text-to-Spatial: Grounding text in positional layout  
    - Spatial-to-Text: Propagating layout context to language
    - Spatial-to-Spatial: Pure geometric relationships
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Separate attention modules for different interaction types
        self.text_to_text_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_spatial_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_to_text_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_to_spatial_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Combination weights
        self.combination_weights = nn.Parameter(torch.ones(4) / 4)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, text_features: torch.Tensor, spatial_features: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None,
                spatial_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Apply disentangled attention.
        
        Args:
            text_features: Text features [B, N_text, D]
            spatial_features: Spatial features [B, N_spatial, D]
            text_mask: Optional attention mask for text
            spatial_mask: Optional attention mask for spatial
            
        Returns:
            dict containing fused representations
        """
        # Text-to-Text attention (intra-linguistic dependencies)
        text_to_text, _ = self.text_to_text_attn(
            text_features, text_features, text_features,
            key_padding_mask=text_mask
        )
        
        # Text-to-Spatial attention (grounding text in layout)
        text_to_spatial, _ = self.text_to_spatial_attn(
            text_features, spatial_features, spatial_features,
            key_padding_mask=spatial_mask
        )
        
        # Spatial-to-Text attention (layout context to language)
        spatial_to_text, _ = self.spatial_to_text_attn(
            spatial_features, text_features, text_features,
            key_padding_mask=text_mask
        )
        
        # Spatial-to-Spatial attention (geometric relationships)
        spatial_to_spatial, _ = self.spatial_to_spatial_attn(
            spatial_features, spatial_features, spatial_features,
            key_padding_mask=spatial_mask
        )
        
        # Combine text features
        weights = F.softmax(self.combination_weights, dim=0)
        fused_text = (
            weights[0] * text_to_text + 
            weights[1] * text_to_spatial
        )
        fused_text = self.norm(fused_text + text_features)
        
        # Combine spatial features
        fused_spatial = (
            weights[2] * spatial_to_text + 
            weights[3] * spatial_to_spatial
        )
        fused_spatial = self.norm(fused_spatial + spatial_features)
        
        return {
            'fused_text': fused_text,
            'fused_spatial': fused_spatial,
            'attention_components': {
                'text_to_text': text_to_text,
                'text_to_spatial': text_to_spatial, 
                'spatial_to_text': spatial_to_text,
                'spatial_to_spatial': spatial_to_spatial
            }
        }


class MultiModalSpatialFusion(nn.Module):
    """
    Multi-Modal Spatial Fusion module that combines:
    - Graph features from spatial reasoning
    - Integrated memory and question representations
    - Compressed visual features
    - Spatial layout information
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fusion_dim = config.fusion_dim
        self.num_heads = config.num_attention_heads
        self.num_layers = config.num_fusion_layers
        self.dropout = config.dropout
        self.use_disentangled = config.use_disentangled_attention
        
        # Input projections to common dimension
        self.graph_proj = nn.Linear(768, self.fusion_dim)  # Assuming graph features are 768-dim
        self.memory_proj = nn.Linear(512, self.fusion_dim)  # Memory features are 512-dim  
        self.visual_proj = nn.Linear(768, self.fusion_dim)  # Visual features are 768-dim
        self.question_proj = nn.Linear(512, self.fusion_dim)  # Question features are 512-dim
        
        # Positional encoding for spatial relationships
        self.spatial_pos_encoding = nn.Parameter(
            torch.randn(1, 1000, self.fusion_dim) * 0.1
        )
        
        # Disentangled attention layers
        if self.use_disentangled:
            self.disentangled_layers = nn.ModuleList([
                DisentangledAttention(
                    embed_dim=self.fusion_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout
                )
                for _ in range(self.num_layers)
            ])
        
        # Standard multi-head attention for fallback
        self.cross_modal_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.fusion_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                batch_first=True
            )
            for _ in range(self.num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.fusion_dim) for _ in range(self.num_layers * 2)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim * 4),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.fusion_dim * 4, self.fusion_dim)
            )
            for _ in range(self.num_layers)
        ])
        
        # Final fusion layers
        self.final_fusion = nn.Sequential(
            nn.Linear(self.fusion_dim * 4, self.fusion_dim * 2),  # Combine all modalities
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )
        
        # Modality-specific gates
        self.modality_gates = nn.ModuleDict({
            'visual': nn.Sequential(nn.Linear(self.fusion_dim, 1), nn.Sigmoid()),
            'textual': nn.Sequential(nn.Linear(self.fusion_dim, 1), nn.Sigmoid()),
            'spatial': nn.Sequential(nn.Linear(self.fusion_dim, 1), nn.Sigmoid()),
            'memory': nn.Sequential(nn.Linear(self.fusion_dim, 1), nn.Sigmoid())
        })
        
    def add_positional_encoding(self, features: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add spatial positional encoding to features."""
        batch_size, seq_len, _ = features.shape
        
        if positions is not None:
            # Use provided positions
            pos_enc = self.spatial_pos_encoding[:, positions, :]
        else:
            # Use sequential positions
            pos_enc = self.spatial_pos_encoding[:, :seq_len, :]
        
        return features + pos_enc.expand(batch_size, -1, -1)
    
    def prepare_modality_features(self, graph_features: torch.Tensor,
                                integrated_memory: torch.Tensor,
                                compressed_visual: torch.Tensor,
                                question_processed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Prepare and project all modality features to common dimension.
        
        Args:
            graph_features: Graph features [B, N_graph, D_graph]
            integrated_memory: Memory features [B, D_memory]
            compressed_visual: Visual features [B, N_visual, D_visual] 
            question_processed: Question features [B, D_question]
            
        Returns:
            dict of projected features
        """
        # Project to common fusion dimension
        graph_proj = self.graph_proj(graph_features)
        
        # Handle memory features (expand if needed)
        if integrated_memory.dim() == 2:
            memory_proj = self.memory_proj(integrated_memory).unsqueeze(1)
        else:
            memory_proj = self.memory_proj(integrated_memory)
            
        visual_proj = self.visual_proj(compressed_visual)
        
        # Handle question features (expand if needed)
        if question_processed.dim() == 2:
            question_proj = self.question_proj(question_processed).unsqueeze(1)
        else:
            question_proj = self.question_proj(question_processed)
        
        # Add positional encoding
        graph_proj = self.add_positional_encoding(graph_proj)
        visual_proj = self.add_positional_encoding(visual_proj)
        
        return {
            'graph': graph_proj,
            'memory': memory_proj,
            'visual': visual_proj,
            'question': question_proj
        }
    
    def apply_disentangled_fusion(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply disentangled attention fusion across modalities."""
        # Combine textual modalities (memory + question)
        textual_features = torch.cat([
            modality_features['memory'], 
            modality_features['question']
        ], dim=1)
        
        # Combine spatial modalities (graph + visual)
        spatial_features = torch.cat([
            modality_features['graph'],
            modality_features['visual']  
        ], dim=1)
        
        # Apply disentangled attention layers
        for i, disentangled_layer in enumerate(self.disentangled_layers):
            fusion_result = disentangled_layer(textual_features, spatial_features)
            
            textual_features = fusion_result['fused_text']
            spatial_features = fusion_result['fused_spatial']
            
            # Feed-forward networks
            textual_ffn = self.ffn_layers[i](textual_features)
            spatial_ffn = self.ffn_layers[i](spatial_features)
            
            textual_features = self.layer_norms[i*2](textual_features + textual_ffn)
            spatial_features = self.layer_norms[i*2+1](spatial_features + spatial_ffn)
        
        # Global pooling
        textual_global = textual_features.mean(dim=1)
        spatial_global = spatial_features.mean(dim=1)
        
        return torch.cat([textual_global, spatial_global], dim=-1)
    
    def apply_standard_fusion(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply standard cross-modal attention fusion."""
        # Concatenate all modalities
        all_features = torch.cat([
            modality_features['graph'],
            modality_features['memory'],
            modality_features['visual'], 
            modality_features['question']
        ], dim=1)
        
        # Apply cross-modal attention layers
        fused = all_features
        for i, (attention_layer, ffn_layer) in enumerate(zip(
            self.cross_modal_attention, self.ffn_layers
        )):
            # Self-attention
            attended, _ = attention_layer(fused, fused, fused)
            fused = self.layer_norms[i*2](attended + fused)
            
            # Feed-forward
            ffn_output = ffn_layer(fused)
            fused = self.layer_norms[i*2+1](ffn_output + fused)
        
        # Global pooling
        fused_global = fused.mean(dim=1)
        
        return fused_global
    
    def apply_modality_gating(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply modality-specific gating mechanisms."""
        gated_features = {}
        
        for modality, feat in features.items():
            if feat.dim() == 3:  # [B, N, D]
                global_feat = feat.mean(dim=1)  # [B, D]
            else:
                global_feat = feat
            
            gate = self.modality_gates[modality](global_feat)
            gated_features[modality] = feat * gate.unsqueeze(-1)
            if feat.dim() == 3:
                gated_features[modality] = gated_features[modality]
        
        return gated_features
    
    def forward(self, graph_features: torch.Tensor,
                integrated_memory: torch.Tensor,
                compressed_visual: torch.Tensor,
                question_processed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Multi-Modal Spatial Fusion.
        
        Args:
            graph_features: Features from graph reasoning [B, N_graph, D]
            integrated_memory: Memory-integrated features [B, D] or [B, N, D]  
            compressed_visual: Compressed visual features [B, N_visual, D]
            question_processed: Processed question features [B, D] or [B, N, D]
            
        Returns:
            dict containing:
                - fused_representation: Final fused multimodal representation
                - modality_contributions: Individual modality contributions
                - attention_weights: Attention visualization info
        """
        # Prepare modality features
        modality_features = self.prepare_modality_features(
            graph_features, integrated_memory, compressed_visual, question_processed
        )
        
        # Apply modality gating
        gated_features = self.apply_modality_gating(modality_features)
        
        # Apply fusion strategy
        if self.use_disentangled:
            fused_global = self.apply_disentangled_fusion(gated_features)
        else:
            fused_global = self.apply_standard_fusion(gated_features)
        
        # Final fusion layer
        if fused_global.dim() == 2 and fused_global.shape[-1] != self.fusion_dim * 2:
            # Need to concatenate all modality global representations
            modality_globals = []
            for modality, feat in gated_features.items():
                if feat.dim() == 3:
                    modality_globals.append(feat.mean(dim=1))
                else:
                    modality_globals.append(feat.squeeze(1) if feat.dim() == 3 else feat)
            
            fused_global = torch.cat(modality_globals, dim=-1)
        
        final_representation = self.final_fusion(fused_global)
        
        return {
            'fused_representation': final_representation,
            'modality_features': modality_features,
            'gated_features': gated_features,
            'fusion_info': {
                'fusion_strategy': 'disentangled' if self.use_disentangled else 'standard',
                'num_layers': self.num_layers,
                'fusion_dim': self.fusion_dim
            }
        }


def create_fusion_module(config):
    """Factory function to create multi-modal fusion module."""
    return MultiModalSpatialFusion(config)
