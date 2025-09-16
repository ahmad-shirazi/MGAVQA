"""
Main MGA-VQA Model Implementation.
Orchestrates all components of the Multi-Modal Graph-Augmented Visual Question Answering system.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mga_vqa.modules.visual_encoder import GemmaVLMVisualEncoder
from mga_vqa.modules.spatial_graph import SpatialGraphConstructor, GraphNeuralNetwork
from mga_vqa.modules.question_processor import MemoryAugmentedQuestionProcessor
from mga_vqa.modules.compressor import QuestionGuidedCompressor
from mga_vqa.modules.fusion import MultiModalSpatialFusion
from mga_vqa.config import MGAVQAConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


class MGA_VQA(nn.Module):
    """
    Main MGA-VQA model with Gemma-3-8B backbone that integrates all components:
    1. Token-Level Visual Encoding
    2. Spatial Graph Construction and Reasoning 
    3. Memory-Augmented Question Processing (with Gemma-3-8B)
    4. Question-Guided Compression
    5. Multi-Modal Spatial Fusion
    6. Answer Prediction and Bounding Box Localization
    """
    
    def __init__(self, config: MGAVQAConfig):
        super().__init__()
        self.config = config
        
        # Gemma-3 VLM and Language backbone integration
        self.vlm_backbone = config.vlm_backbone  # PaliGemma for visual processing
        self.language_backbone = config.language_backbone  # Gemma-3-8B for language
        
        self.backbone_tokenizer = AutoTokenizer.from_pretrained(config.language_backbone)
        if self.backbone_tokenizer.pad_token is None:
            self.backbone_tokenizer.pad_token = self.backbone_tokenizer.eos_token
        
        # Initialize all sub-modules with Gemma-3 VLM
        self.visual_encoder = GemmaVLMVisualEncoder(config.visual_encoder)
        self.graph_constructor = SpatialGraphConstructor(config.graph)
        self.graph_reasoner = GraphNeuralNetwork(config.graph)
        self.question_processor = MemoryAugmentedQuestionProcessor(config.memory)
        self.compressor = QuestionGuidedCompressor(config.compression)
        self.fusion_module = MultiModalSpatialFusion(config.fusion)
        
        # Gemma-3-8B enhanced prediction heads
        backbone_hidden_size = 4096  # Typical for Gemma-3-8B models
        self.answer_head = GemmaAnswerHead(
            input_dim=config.fusion.fusion_dim,
            backbone_dim=backbone_hidden_size,
            vocab_size=config.answer_vocab_size,
            max_length=config.max_answer_length,
            backbone_model=config.language_backbone  # Use language backbone for answer generation
        )
        
        self.bbox_head = BoundingBoxLocalizationHead(
            input_dim=config.fusion.fusion_dim,
            hidden_dim=512
        )
        
        # Loss functions
        self.answer_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.bbox_criterion = nn.SmoothL1Loss()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, mean=0.0, std=0.02)
    
    def forward(self, 
                image: torch.Tensor,
                question_text: List[str], 
                ocr_results: List[List[Dict]],
                image_sizes: List[Tuple[int, int]],
                answer_labels: Optional[torch.Tensor] = None,
                bbox_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through MGA-VQA pipeline.
        
        Args:
            image: Document image tensor [B, C, H, W]
            question_text: List of question strings (length B)
            ocr_results: List of OCR results for each image (length B)
                        Each element is a list of dicts with 'text' and 'bbox' keys
            image_sizes: List of (width, height) tuples for each image
            answer_labels: Optional answer labels for training [B, max_length]
            bbox_targets: Optional bounding box targets for training [B, 4]
            
        Returns:
            dict containing:
                - answer_logits: Answer prediction logits
                - bbox_predictions: Bounding box predictions
                - loss: Total loss (if labels provided)
                - intermediate_outputs: All intermediate representations
        """
        batch_size = image.shape[0]
        device = image.device
        
        # 1. Token-Level Visual Encoding
        visual_outputs = self.visual_encoder(image)
        visual_features = visual_outputs['visual_features']  # [B, N, D]
        
        # 2. Question Processing with Memory Augmentation
        question_outputs = self.question_processor(question_text)
        question_processed = question_outputs['question_processed']  # [B, D]
        integrated_memory = question_outputs['integrated_memory']    # [B, D]
        
        # 3. Question-Guided Compression
        compression_outputs = self.compressor(
            visual_features, question_processed, use_hard_selection=True
        )
        compressed_visual = compression_outputs['compressed_tokens']  # [B, k, D]
        
        # 4. Spatial Graph Construction and Reasoning (per sample in batch)
        graph_features_batch = []
        for i in range(batch_size):
            # Construct graph for each sample
            graph_data = self.graph_constructor(
                ocr_results[i], 
                visual_features[i:i+1], 
                image_sizes[i]
            )
            
            # Apply graph reasoning
            graph_outputs = self.graph_reasoner(graph_data)
            graph_features_batch.append(graph_outputs['node_embeddings'])
        
        # Pad graph features to same length
        max_nodes = max(feat.shape[0] for feat in graph_features_batch)
        if max_nodes == 0:
            max_nodes = 1  # Avoid empty tensors
        
        padded_graph_features = []
        for feat in graph_features_batch:
            if feat.shape[0] == 0:
                # Handle empty graphs
                padded_feat = torch.zeros(1, feat.shape[-1] if feat.numel() > 0 else self.config.graph.node_dim).to(device)
            else:
                pad_size = max_nodes - feat.shape[0]
                if pad_size > 0:
                    padding = torch.zeros(pad_size, feat.shape[-1]).to(device)
                    padded_feat = torch.cat([feat, padding], dim=0)
                else:
                    padded_feat = feat
            padded_graph_features.append(padded_feat)
        
        graph_features = torch.stack(padded_graph_features)  # [B, max_nodes, D]
        
        # 5. Multi-Modal Spatial Fusion
        fusion_outputs = self.fusion_module(
            graph_features=graph_features,
            integrated_memory=integrated_memory,
            compressed_visual=compressed_visual,
            question_processed=question_processed
        )
        fused_representation = fusion_outputs['fused_representation']  # [B, D]
        
        # 6. Answer Prediction and Bounding Box Localization
        answer_logits = self.answer_head(fused_representation)         # [B, vocab_size, max_length]
        bbox_predictions = self.bbox_head(fused_representation)        # [B, 4]
        
        # Compute loss if targets are provided
        total_loss = None
        answer_loss = None
        bbox_loss = None
        
        if answer_labels is not None:
            answer_loss = self.compute_answer_loss(answer_logits, answer_labels)
            total_loss = answer_loss
        
        if bbox_targets is not None:
            bbox_loss = self.compute_bbox_loss(bbox_predictions, bbox_targets)
            if total_loss is not None:
                total_loss = total_loss + bbox_loss
            else:
                total_loss = bbox_loss
        
        return {
            'answer_logits': answer_logits,
            'bbox_predictions': bbox_predictions,
            'total_loss': total_loss,
            'answer_loss': answer_loss,
            'bbox_loss': bbox_loss,
            'intermediate_outputs': {
                'visual_features': visual_features,
                'compressed_visual': compressed_visual,
                'question_processed': question_processed,
                'integrated_memory': integrated_memory,
                'graph_features': graph_features,
                'fused_representation': fused_representation,
                'compression_info': compression_outputs['compression_info']
            }
        }
    
    def compute_answer_loss(self, answer_logits: torch.Tensor, 
                          answer_labels: torch.Tensor) -> torch.Tensor:
        """Compute answer prediction loss."""
        # Reshape for cross entropy: [B*max_length, vocab_size] and [B*max_length]
        logits_flat = answer_logits.transpose(1, 2).contiguous().view(-1, answer_logits.size(1))
        labels_flat = answer_labels.view(-1)
        
        answer_loss = self.answer_criterion(logits_flat, labels_flat)
        return answer_loss
    
    def compute_bbox_loss(self, bbox_predictions: torch.Tensor,
                         bbox_targets: torch.Tensor) -> torch.Tensor:
        """Compute bounding box localization loss."""
        bbox_loss = self.bbox_criterion(bbox_predictions, bbox_targets)
        return bbox_loss
    
    def generate_answer(self, answer_logits: torch.Tensor, 
                       max_length: Optional[int] = None) -> List[List[int]]:
        """Generate answer sequences from logits."""
        if max_length is None:
            max_length = self.config.max_answer_length
        
        batch_size, vocab_size, seq_length = answer_logits.shape
        
        # Simple greedy decoding
        predicted_ids = answer_logits.argmax(dim=1)  # [B, max_length]
        
        # Convert to list of sequences
        answer_sequences = []
        for i in range(batch_size):
            sequence = predicted_ids[i].tolist()
            # Remove padding tokens (assuming 0 is padding)
            sequence = [token_id for token_id in sequence if token_id != 0]
            answer_sequences.append(sequence)
        
        return answer_sequences
    
    def predict(self, image: torch.Tensor,
                question_text: List[str],
                ocr_results: List[List[Dict]], 
                image_sizes: List[Tuple[int, int]]) -> Dict[str, Union[List, torch.Tensor]]:
        """
        Prediction interface for inference.
        
        Args:
            image: Document image tensor [B, C, H, W] 
            question_text: List of question strings
            ocr_results: List of OCR results for each image
            image_sizes: List of (width, height) tuples
            
        Returns:
            dict containing predictions and confidence scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                image=image,
                question_text=question_text, 
                ocr_results=ocr_results,
                image_sizes=image_sizes
            )
            
            # Generate answer sequences
            answer_sequences = self.generate_answer(outputs['answer_logits'])
            
            # Get confidence scores
            answer_probs = F.softmax(outputs['answer_logits'], dim=1)
            answer_confidence = answer_probs.max(dim=1)[0].mean(dim=1)
            
            bbox_predictions = outputs['bbox_predictions']
            
            return {
                'answer_sequences': answer_sequences,
                'answer_confidence': answer_confidence.tolist(),
                'bbox_predictions': bbox_predictions,
                'intermediate_outputs': outputs['intermediate_outputs']
            }


class GemmaAnswerHead(nn.Module):
    """Gemma-3-8B enhanced answer prediction head for generating textual answers."""
    
    def __init__(self, input_dim: int, backbone_dim: int, vocab_size: int, 
                 max_length: int, backbone_model: str):
        super().__init__()
        self.input_dim = input_dim
        self.backbone_dim = backbone_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Gemma backbone integration (frozen for efficiency)
        self.gemma_tokenizer = AutoTokenizer.from_pretrained(backbone_model)
        if self.gemma_tokenizer.pad_token is None:
            self.gemma_tokenizer.pad_token = self.gemma_tokenizer.eos_token
        
        # Feature adaptation layers
        self.feature_adapter = nn.Sequential(
            nn.Linear(input_dim, backbone_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(backbone_dim)
        )
        
        # Cross-attention with Gemma features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=backbone_dim,
            num_heads=16,
            dropout=0.1,
            batch_first=True
        )
        
        # Answer generation layers inspired by Gemma architecture
        self.answer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=backbone_dim,
                nhead=16,
                dim_feedforward=backbone_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(3)
        ])
        
        # Final projection to vocabulary
        self.output_projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(backbone_dim // 2, vocab_size)
        )
        
        # Positional encoding for answer generation
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_length, backbone_dim) * 0.02
        )
    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Generate answer logits using Gemma-enhanced architecture.
        
        Args:
            fused_features: Fused multimodal features [B, D]
            
        Returns:
            answer_logits: [B, vocab_size, max_length]
        """
        batch_size = fused_features.shape[0]
        
        # Adapt features to Gemma dimension
        adapted_features = self.feature_adapter(fused_features)  # [B, backbone_dim]
        adapted_features = adapted_features.unsqueeze(1)  # [B, 1, backbone_dim]
        
        # Create answer sequence embeddings
        answer_embeds = self.pos_encoding.expand(batch_size, -1, -1)  # [B, max_length, backbone_dim]
        
        # Apply cross-attention between multimodal features and answer positions
        attended_features, _ = self.cross_attention(
            answer_embeds, adapted_features, adapted_features
        )  # [B, max_length, backbone_dim]
        
        # Apply transformer decoder layers
        hidden_states = attended_features
        for layer in self.answer_layers:
            hidden_states = layer(hidden_states, adapted_features)
        
        # Project to vocabulary
        answer_logits = self.output_projection(hidden_states)  # [B, max_length, vocab_size]
        answer_logits = answer_logits.transpose(1, 2)  # [B, vocab_size, max_length]
        
        return answer_logits


class AnswerPredictionHead(nn.Module):
    """Standard answer prediction head for generating textual answers."""
    
    def __init__(self, input_dim: int, vocab_size: int, max_length: int):
        super().__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Answer generation layers
        self.answer_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, vocab_size * max_length)
        )
    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Generate answer logits.
        
        Args:
            fused_features: Fused multimodal features [B, D]
            
        Returns:
            answer_logits: [B, vocab_size, max_length]
        """
        batch_size = fused_features.shape[0]
        
        # Project to answer space
        answer_flat = self.answer_projection(fused_features)  # [B, vocab_size * max_length]
        
        # Reshape to [B, vocab_size, max_length]
        answer_logits = answer_flat.view(batch_size, self.vocab_size, self.max_length)
        
        return answer_logits


class BoundingBoxLocalizationHead(nn.Module):
    """Bounding box localization head for answer grounding."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Bounding box regression layers
        self.bbox_regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4),  # x1, y1, x2, y2
            nn.Sigmoid()  # Normalize to [0, 1]
        )
    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Predict bounding box coordinates.
        
        Args:
            fused_features: Fused multimodal features [B, D]
            
        Returns:
            bbox_predictions: [B, 4] normalized coordinates
        """
        bbox_predictions = self.bbox_regressor(fused_features)
        return bbox_predictions


def create_mga_vqa_model(config: MGAVQAConfig) -> MGA_VQA:
    """Factory function to create MGA-VQA model."""
    return MGA_VQA(config)
