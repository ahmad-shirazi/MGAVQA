"""
Gemma-3 VLM Token-Level Visual Encoder for MGA-VQA.
Implements token-level visual feature extraction using Gemma-3 Vision-Language Model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import torchvision.transforms as transforms
from transformers import (
    AutoProcessor, 
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    SiglipVisionModel,
    GemmaTokenizer,
    AutoModel
)
import math
import numpy as np


class MultiScaleTokenProcessor(nn.Module):
    """Multi-scale token processing for Gemma-3 VLM features."""
    
    def __init__(self, input_dim: int, scales: List[int] = [14, 28, 56], dropout: float = 0.1):
        super().__init__()
        self.scales = scales
        self.input_dim = input_dim
        
        # Scale-specific processing layers
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(input_dim)
            )
            for _ in scales
        ])
        
        # Cross-scale attention
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        """
        Process tokens at multiple scales.
        
        Args:
            token_features: Input token features [B, N, D]
        Returns:
            processed_features: Multi-scale processed features [B, N, D]
        """
        batch_size, num_tokens, _ = token_features.shape
        
        # Process at different scales (simulate by different transformations)
        scale_features = []
        for processor in self.scale_processors:
            scale_feature = processor(token_features)
            scale_features.append(scale_feature)
        
        # Concatenate and apply cross-scale attention
        all_scales = torch.cat(scale_features, dim=1)  # [B, N*num_scales, D]
        
        # Apply cross-attention
        attended_features, _ = self.cross_scale_attention(
            token_features, all_scales, all_scales
        )
        
        # Residual connection and normalization
        output = self.norm(attended_features + token_features)
        
        return output


class DocumentSpecificAdapter(nn.Module):
    """Document-specific adaptation layers for visual features."""
    
    def __init__(self, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Document-specific adaptation layers
        self.adaptation_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Text-aware visual enhancement
        self.text_aware_enhancement = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Apply document-specific adaptations.
        
        Args:
            visual_features: Input visual features [B, N, D]
        Returns:
            adapted_features: Document-adapted features [B, N, D]
        """
        adapted = visual_features
        
        # Apply adaptation layers
        for layer in self.adaptation_layers:
            adapted = layer(adapted)
        
        # Apply text-aware enhancement
        enhanced = self.text_aware_enhancement(adapted)
        adapted = adapted + enhanced
        
        return adapted


class SpatialPositionalEncoding(nn.Module):
    """Spatial positional encoding for token-level features."""
    
    def __init__(self, hidden_dim: int, max_seq_len: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable positional embeddings
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )
        
        # 2D spatial encoding
        self.spatial_proj = nn.Linear(4, hidden_dim)  # x, y, width, height
        
    def forward(self, features: torch.Tensor, spatial_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add spatial positional encoding.
        
        Args:
            features: Input features [B, N, D]
            spatial_info: Optional spatial information [B, N, 4]
        Returns:
            features_with_pos: Features with positional encoding [B, N, D]
        """
        batch_size, seq_len, _ = features.shape
        
        # Add learnable positional embeddings
        pos_enc = self.pos_embeddings[:, :seq_len, :]
        features = features + pos_enc
        
        # Add spatial information if provided
        if spatial_info is not None:
            spatial_enc = self.spatial_proj(spatial_info)
            features = features + spatial_enc
        
        return features


class AttentionTokenPooler(nn.Module):
    """Attention-based token pooling."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pool tokens using attention weights.
        
        Args:
            features: Input features [B, N, D]
        Returns:
            pooled: Pooled features [B, D]
        """
        # Compute attention weights
        attention_weights = self.attention(features)  # [B, N, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        pooled = (features * attention_weights).sum(dim=1)  # [B, D]
        
        return pooled


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial relationships."""
    
    def __init__(self, embed_dim: int, max_height: int = 1000, max_width: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create 2D positional encoding
        pe_h = torch.zeros(max_height, embed_dim // 2)
        pe_w = torch.zeros(max_width, embed_dim // 2)
        
        position_h = torch.arange(0, max_height).unsqueeze(1).float()
        position_w = torch.arange(0, max_width).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim // 2, 2).float() *
                           -(math.log(10000.0) / (embed_dim // 2)))
        
        pe_h[:, 0::2] = torch.sin(position_h * div_term)
        pe_h[:, 1::2] = torch.cos(position_h * div_term)
        pe_w[:, 0::2] = torch.sin(position_w * div_term)
        pe_w[:, 1::2] = torch.cos(position_w * div_term)
        
        self.register_buffer('pe_h', pe_h)
        self.register_buffer('pe_w', pe_w)
    
    def forward(self, height_positions: torch.Tensor, width_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            height_positions: [B, N] height indices
            width_positions: [B, N] width indices
        Returns:
            pos_enc: [B, N, D] positional encodings
        """
        batch_size, num_patches = height_positions.shape
        
        h_enc = self.pe_h[height_positions]  # [B, N, D//2]
        w_enc = self.pe_w[width_positions]   # [B, N, D//2]
        
        pos_enc = torch.cat([h_enc, w_enc], dim=-1)  # [B, N, D]
        return pos_enc


class GemmaVLMVisualEncoder(nn.Module):
    """
    Gemma-3 Vision-Language Model for Token-Level Visual Feature Extraction.
    Uses PaliGemma (Gemma-3 VLM) for sophisticated visual understanding of document images.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.vision_hidden_dim = config.vision_hidden_dim
        self.image_token_length = config.image_token_length
        
        # Gemma-3 VLM Components (PaliGemma)
        self.vlm_processor = PaliGemmaProcessor.from_pretrained(config.vlm_model_name)
        self.vlm_model = PaliGemmaForConditionalGeneration.from_pretrained(
            config.vlm_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Extract vision encoder from PaliGemma for token-level features
        self.vision_tower = self.vlm_model.vision_tower
        
        # Freeze VLM weights initially (can be unfrozen during training)
        for param in self.vlm_model.parameters():
            param.requires_grad = False
        
        # Vision feature projection to desired hidden dimension
        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        # Multi-scale token processing
        self.token_scale_processor = MultiScaleTokenProcessor(
            input_dim=self.hidden_dim,
            scales=config.patch_sizes,
            dropout=config.dropout
        )
        
        # Token-level refinement with document-specific adaptations
        self.document_adapter = DocumentSpecificAdapter(
            hidden_dim=self.hidden_dim,
            num_layers=3,
            dropout=config.dropout
        )
        
        # Positional encoding for spatial relationships
        self.spatial_pos_encoding = SpatialPositionalEncoding(
            hidden_dim=self.hidden_dim,
            max_seq_len=self.image_token_length
        )
        
        # Token pooling strategy
        if config.token_pooling_strategy == "attention":
            self.token_pooler = AttentionTokenPooler(self.hidden_dim)
        elif config.token_pooling_strategy == "mean":
            self.token_pooler = lambda x: x.mean(dim=1, keepdim=True)
        else:  # max
            self.token_pooler = lambda x: x.max(dim=1, keepdim=True)[0]
    
    def extract_visual_tokens_from_vlm(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract token-level visual features using Gemma-3 VLM.
        
        Args:
            image: Input image tensor [B, C, H, W]
        Returns:
            visual_tokens: Token-level visual features [B, N, D]
        """
        # Convert tensor to PIL Images for processor
        if image.dim() == 4:
            batch_size = image.shape[0]
            pil_images = []
            
            for i in range(batch_size):
                # Denormalize if needed
                img = image[i]
                if img.min() < 0:  # Assume normalized
                    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                          torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                
                img = torch.clamp(img, 0, 1)
                img_np = (img.permute(1, 2, 0) * 255).byte().cpu().numpy()
                
                from PIL import Image
                pil_img = Image.fromarray(img_np)
                pil_images.append(pil_img)
        else:
            batch_size = 1
            pil_images = [image]
        
        # Process images through VLM vision tower
        with torch.no_grad():
            vision_outputs = []
            for pil_img in pil_images:
                # Use a dummy prompt for visual feature extraction
                inputs = self.vlm_processor(
                    text=["describe this image"],
                    images=[pil_img],
                    return_tensors="pt"
                )
                
                # Move to device
                device = next(self.vlm_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Extract vision features before language processing
                pixel_values = inputs["pixel_values"]
                vision_features = self.vision_tower(pixel_values)
                
                # Get the last hidden state which contains token-level features
                if hasattr(vision_features, 'last_hidden_state'):
                    tokens = vision_features.last_hidden_state
                else:
                    tokens = vision_features
                
                vision_outputs.append(tokens)
            
            # Stack batch
            visual_tokens = torch.stack(vision_outputs, dim=0)
            if visual_tokens.dim() == 4:  # [B, 1, N, D]
                visual_tokens = visual_tokens.squeeze(1)  # [B, N, D]
        
        return visual_tokens
    
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image for Gemma-3 VLM."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Ensure image is in the right format for VLM
        # Most VLMs expect specific input sizes
        target_size = 224  # Common VLM input size
        
        if image.shape[-1] != target_size or image.shape[-2] != target_size:
            image = F.interpolate(
                image, 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        return image
    
    def create_position_indices(self, patch_positions: List[Tuple[int, int]], 
                              batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create position indices for patches."""
        all_h_indices = []
        all_w_indices = []
        
        for patch_h, patch_w in patch_positions:
            h_indices = torch.arange(patch_h).repeat_interleave(patch_w)
            w_indices = torch.arange(patch_w).repeat(patch_h)
            all_h_indices.append(h_indices)
            all_w_indices.append(w_indices)
        
        h_indices = torch.cat(all_h_indices).unsqueeze(0).repeat(batch_size, 1)
        w_indices = torch.cat(all_w_indices).unsqueeze(0).repeat(batch_size, 1)
        
        return h_indices.to(next(self.parameters()).device), w_indices.to(next(self.parameters()).device)
    
    def align_with_text_regions(self, visual_features: torch.Tensor, 
                               text_bboxes: Optional[List[List[float]]] = None) -> torch.Tensor:
        """Align visual features with text regions if provided."""
        if text_bboxes is None:
            return visual_features
        
        # Apply alignment layers for better text-visual correspondence
        aligned_features = visual_features
        for layer in self.alignment_layers:
            aligned_features = layer(aligned_features)
        
        return aligned_features
    
    def forward(self, image: torch.Tensor, 
                text_bboxes: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        Forward pass through Gemma-3 VLM Token-Level Visual Encoder.
        
        Args:
            image: Input document image [B, C, H, W]
            text_bboxes: Optional text bounding boxes for spatial alignment
            
        Returns:
            dict containing:
                - visual_features: Token-level visual features [B, N, D]
                - vlm_tokens: Raw VLM token features [B, N, vision_dim]  
                - spatial_features: Spatially-aware features [B, N, D]
                - document_adapted: Document-specific adapted features [B, N, D]
        """
        # Preprocess image for VLM
        processed_image = self.preprocess_image(image)
        batch_size = processed_image.shape[0]
        
        # Extract token-level visual features using Gemma-3 VLM
        vlm_visual_tokens = self.extract_visual_tokens_from_vlm(processed_image)  # [B, N, vision_dim]
        
        # Project VLM features to target hidden dimension
        projected_tokens = self.vision_proj(vlm_visual_tokens)  # [B, N, hidden_dim]
        
        # Apply multi-scale token processing
        multi_scale_tokens = self.token_scale_processor(projected_tokens)  # [B, N, hidden_dim]
        
        # Add spatial positional encoding
        if text_bboxes is not None:
            # Create spatial information from bounding boxes
            spatial_info = self.create_spatial_info_from_bboxes(text_bboxes, batch_size)
            spatial_tokens = self.spatial_pos_encoding(multi_scale_tokens, spatial_info)
        else:
            spatial_tokens = self.spatial_pos_encoding(multi_scale_tokens)
        
        # Apply document-specific adaptations
        document_adapted = self.document_adapter(spatial_tokens)  # [B, N, hidden_dim]
        
        # Final token-level features
        visual_features = document_adapted
        
        return {
            'visual_features': visual_features,
            'vlm_tokens': vlm_visual_tokens,
            'spatial_features': spatial_tokens,
            'document_adapted': document_adapted,
            'token_count': visual_features.shape[1]
        }
    
    def create_spatial_info_from_bboxes(self, text_bboxes: List[List[float]], 
                                      batch_size: int) -> torch.Tensor:
        """
        Create spatial information tensor from text bounding boxes.
        
        Args:
            text_bboxes: List of bounding boxes per batch item
            batch_size: Batch size
            
        Returns:
            spatial_info: Spatial information tensor [B, N, 4]
        """
        device = next(self.parameters()).device
        max_boxes = max(len(boxes) for boxes in text_bboxes) if text_bboxes else 1
        
        spatial_info = torch.zeros(batch_size, max_boxes, 4).to(device)
        
        for i, boxes in enumerate(text_bboxes):
            for j, bbox in enumerate(boxes):
                if j < max_boxes:
                    spatial_info[i, j] = torch.tensor(bbox, dtype=torch.float32)
        
        return spatial_info
    
    def get_token_level_features(self, image: torch.Tensor, 
                               num_tokens: Optional[int] = None) -> torch.Tensor:
        """
        Get specific number of token-level features.
        
        Args:
            image: Input image [B, C, H, W]
            num_tokens: Desired number of tokens (None for all)
            
        Returns:
            token_features: Token-level features [B, num_tokens, D]
        """
        outputs = self.forward(image)
        features = outputs['visual_features']
        
        if num_tokens is not None and features.shape[1] != num_tokens:
            if features.shape[1] > num_tokens:
                # Use attention pooling to reduce tokens
                pooled = self.token_pooler(features)
                # Expand to desired number of tokens
                features = pooled.unsqueeze(1).expand(-1, num_tokens, -1)
            else:
                # Pad with zeros if we need more tokens
                padding = torch.zeros(
                    features.shape[0], 
                    num_tokens - features.shape[1], 
                    features.shape[2]
                ).to(features.device)
                features = torch.cat([features, padding], dim=1)
        
        return features


# Alias for backward compatibility  
TokenLevelVisualEncoder = GemmaVLMVisualEncoder


def create_visual_encoder(config):
    """Factory function to create Gemma-3 VLM visual encoder."""
    return GemmaVLMVisualEncoder(config)
