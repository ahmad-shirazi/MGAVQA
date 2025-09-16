"""
MGA-VQA core modules.
"""
from .visual_encoder import TokenLevelVisualEncoder, create_visual_encoder
from .spatial_graph import SpatialGraphConstructor, GraphNeuralNetwork, create_spatial_modules
from .question_processor import MemoryAugmentedQuestionProcessor, create_question_processor
from .compressor import QuestionGuidedCompressor, create_compressor
from .fusion import MultiModalSpatialFusion, create_fusion_module

__all__ = [
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
    'create_fusion_module'
]
