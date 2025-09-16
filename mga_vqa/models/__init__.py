"""
MGA-VQA models and baselines.
"""
from .mga_vqa import MGA_VQA, GemmaAnswerHead, AnswerPredictionHead, BoundingBoxLocalizationHead, create_mga_vqa_model
from .baselines import (
    BaselineLLM, TextOnlyBaseline, MultiModalBaseline, VisionLanguageBaseline,
    BaselineModelFactory, BaselineEvaluator, create_baseline_evaluator
)

__all__ = [
    # Main MGA-VQA model with Gemma-3-8B
    'MGA_VQA',
    'GemmaAnswerHead',
    'AnswerPredictionHead', 
    'BoundingBoxLocalizationHead',
    'create_mga_vqa_model',
    
    # Baseline models for comparison
    'BaselineLLM',
    'TextOnlyBaseline',
    'MultiModalBaseline',
    'VisionLanguageBaseline',
    'BaselineModelFactory',
    'BaselineEvaluator',
    'create_baseline_evaluator'
]
