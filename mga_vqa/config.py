"""
Configuration management for MGA-VQA model.
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch


@dataclass
class VisualEncoderConfig:
    """Configuration for Gemma-3 VLM Token-Level Visual Encoder."""
    # Gemma-3 VLM configuration
    vlm_model_name: str = "google/paligemma-3b-pt-224"  # PaliGemma as Gemma-3 VLM
    vision_model_name: str = "google/siglip-so400m-patch14-384"  # SigLIP vision encoder
    language_model_name: str = "google/gemma-2-2b-it"  # Gemma-2 language component
    
    # Visual processing configuration
    hidden_dim: int = 2048  # Gemma-3 hidden dimension
    vision_hidden_dim: int = 1152  # SigLIP hidden dimension
    num_scales: int = 3
    patch_sizes: List[int] = None
    max_image_size: int = 1024
    image_token_length: int = 256  # Number of image tokens for Gemma-3
    
    # Token-level processing
    use_token_level_features: bool = True
    token_pooling_strategy: str = "attention"  # "mean", "max", "attention"
    num_vision_layers: int = 27  # SigLIP layers
    
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.patch_sizes is None:
            self.patch_sizes = [14, 28, 56]  # Adjusted for SigLIP patch size


@dataclass
class GraphConfig:
    """Configuration for Spatial Graph Construction and Reasoning."""
    node_dim: int = 768
    edge_dim: int = 128
    num_gnn_layers: int = 3
    gnn_hidden_dim: int = 512
    spatial_weight_alpha: float = 0.4
    alignment_weight_beta: float = 0.3
    semantic_weight_gamma: float = 0.3
    max_nodes: int = 256
    dropout: float = 0.1
    use_residual: bool = True


@dataclass
class MemoryConfig:
    """Configuration for Memory-Augmented Question Processing."""
    memory_dim: int = 512
    direct_memory_size: int = 64
    indirect_memory_size: int = 128
    num_attention_heads: int = 8
    question_encoder: str = "google/gemma-2-27b-it"  # Gemma-3-8B for question processing
    max_question_length: int = 128
    dropout: float = 0.1


@dataclass
class CompressionConfig:
    """Configuration for Question-Guided Compression."""
    compression_ratio_min: float = 0.3
    compression_ratio_max: float = 0.8
    similarity_threshold: float = 0.5
    importance_weight: float = 0.5
    adaptive_k: bool = True


@dataclass
class FusionConfig:
    """Configuration for Multi-Modal Spatial Fusion."""
    fusion_dim: int = 768
    num_attention_heads: int = 12
    num_fusion_layers: int = 3
    dropout: float = 0.1
    use_disentangled_attention: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_epochs: int = 50
    warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    val_check_interval: float = 0.5
    early_stopping_patience: int = 5
    save_top_k: int = 3


@dataclass
class BaselineConfig:
    """Configuration for LLM baseline models."""
    # Text-only models
    text_only_models: List[str] = None
    
    # Text + BBox + Image models  
    multimodal_models: List[str] = None
    
    # Image-only models
    vision_language_models: List[str] = None
    
    def __post_init__(self):
        if self.text_only_models is None:
            self.text_only_models = [
                "meta-llama/Llama-2-7b-chat-hf",
                "meta-llama/Meta-Llama-3-8B-Instruct"
            ]
        
        if self.multimodal_models is None:
            self.multimodal_models = [
                "microsoft/LayoutLM-base-uncased",  # LayoutLLM-7B CoT equivalent
                "unstructured-io/Llama-2-7b-document",  # DocLayLLM (Llama2-7B) equivalent
                "unstructured-io/Llama-3-8b-document"   # DocLayLLM (Llama3-7B) equivalent
            ]
        
        if self.vision_language_models is None:
            self.vision_language_models = [
                "microsoft/Phi-3-vision-128k-instruct",  # Phi4-14B equivalent
                "meta-llama/Llama-3.2-11B-Vision-Instruct",
                "mistral-community/pixtral-12b",
                "llava-hf/llava-v1.6-vicuna-13b-hf",  # LLaVA-NeXT-13B
                "llava-hf/llava-onevision-qwen2-7b-ov-hf",  # LLaVA-OneVision-7B
                "Qwen/Qwen2-VL-7B-Instruct",
                "OpenGVLab/InternVL2-8B",
                "mistral-community/pixtral-12b-dlava"  # DLaVA (Pixtral-12B) equivalent
            ]


@dataclass
class MGAVQAConfig:
    """Main configuration for MGA-VQA model."""
    # Component configurations
    visual_encoder: VisualEncoderConfig = None
    graph: GraphConfig = None
    memory: MemoryConfig = None
    compression: CompressionConfig = None
    fusion: FusionConfig = None
    training: TrainingConfig = None
    baselines: BaselineConfig = None
    
    # Model parameters - Gemma-3 VLM as primary backbone
    vlm_backbone: str = "google/paligemma-3b-pt-224"  # Gemma-3 VLM for visual processing
    language_backbone: str = "google/gemma-2-27b-it"  # Gemma-3-8B for language processing
    vocab_size: int = 30000
    answer_vocab_size: int = 10000
    max_answer_length: int = 50
    
    # Device and precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "16-mixed"
    
    # Paths
    data_dir: str = "data/"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    
    def __post_init__(self):
        if self.visual_encoder is None:
            self.visual_encoder = VisualEncoderConfig()
        if self.graph is None:
            self.graph = GraphConfig()
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.compression is None:
            self.compression = CompressionConfig()
        if self.fusion is None:
            self.fusion = FusionConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.baselines is None:
            self.baselines = BaselineConfig()


def get_default_config() -> MGAVQAConfig:
    """Get default configuration for MGA-VQA."""
    return MGAVQAConfig()
