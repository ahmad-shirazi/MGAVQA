"""
Baseline LLM implementations for comparison with MGA-VQA.
Implements all baseline models mentioned in the research paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    LlamaTokenizer, LlamaForCausalLM,
    GemmaTokenizer, GemmaForCausalLM,
    AutoProcessor, AutoModelForVision2Seq,
    Qwen2VLForConditionalGeneration,
    LlavaNextProcessor, LlavaNextForConditionalGeneration
)
from PIL import Image
import numpy as np


class BaselineLLM(nn.Module):
    """Base class for all baseline LLM implementations."""
    
    def __init__(self, model_name: str, device: str = 'cuda'):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.setup_model()
    
    def setup_model(self):
        """Setup tokenizer and model."""
        raise NotImplementedError
    
    def forward(self, *args, **kwargs):
        """Forward pass."""
        raise NotImplementedError
    
    def generate_answer(self, *args, **kwargs):
        """Generate answer for VQA task."""
        raise NotImplementedError


class TextOnlyBaseline(BaselineLLM):
    """
    Text-only baseline models (Llama2-7B-Chat, Llama3-8B-Instruct).
    Only uses OCR text and question, no visual information.
    """
    
    def setup_model(self):
        """Setup text-only model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate_answer(self, question: str, ocr_texts: List[str], 
                       max_new_tokens: int = 50) -> str:
        """
        Generate answer using only question and OCR text.
        
        Args:
            question: Input question
            ocr_texts: List of OCR extracted texts
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer string
        """
        # Create context from OCR texts
        ocr_context = " ".join(ocr_texts)
        
        # Create prompt
        if "llama-2" in self.model_name.lower():
            prompt = f"<s>[INST] Based on the following document text: {ocr_context}\n\nAnswer this question: {question} [/INST]"
        elif "llama-3" in self.model_name.lower():
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nDocument text: {ocr_context}\n\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            prompt = f"Document text: {ocr_context}\nQuestion: {question}\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode answer
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()
        
        return answer


class MultiModalBaseline(BaselineLLM):
    """
    Multi-modal baseline models (LayoutLLM-7B CoT, DocLayLLM variants).
    Uses text + bounding boxes + image information.
    """
    
    def setup_model(self):
        """Setup multi-modal model."""
        if "layoutlm" in self.model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.answer_head = nn.Linear(self.model.config.hidden_size, 10000)
        else:
            # DocLayLLM variants - use Llama backbone
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
    
    def generate_answer(self, question: str, image: torch.Tensor,
                       ocr_results: List[Dict], max_new_tokens: int = 50) -> str:
        """
        Generate answer using question, image, and OCR with bounding boxes.
        
        Args:
            question: Input question
            image: Document image tensor
            ocr_results: List of OCR results with text and bbox
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer string
        """
        # Create structured context with bounding boxes
        structured_context = ""
        for ocr_item in ocr_results:
            text = ocr_item['text']
            bbox = ocr_item['bbox']
            structured_context += f"Text: {text} | BBox: {bbox}\n"
        
        # Create prompt with layout information
        prompt = f"""Document layout information:
{structured_context}

Question: {question}
Answer:"""
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()
        
        return answer


class VisionLanguageBaseline(BaselineLLM):
    """
    Vision-language baseline models (Phi4-14B, Llama3.2-11B, Pixtral-12B, etc.).
    Uses image and question directly.
    """
    
    def setup_model(self):
        """Setup vision-language model."""
        model_type = self.get_model_type()
        
        if model_type == "llava":
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif model_type == "qwen":
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif model_type == "phi":
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # Generic vision-language model
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
    
    def get_model_type(self) -> str:
        """Determine model type from model name."""
        name_lower = self.model_name.lower()
        if "llava" in name_lower:
            return "llava"
        elif "qwen" in name_lower:
            return "qwen"
        elif "phi" in name_lower:
            return "phi"
        elif "pixtral" in name_lower:
            return "pixtral"
        elif "internvl" in name_lower:
            return "internvl"
        else:
            return "generic"
    
    def generate_answer(self, question: str, image: Union[torch.Tensor, Image.Image],
                       max_new_tokens: int = 50) -> str:
        """
        Generate answer using question and image.
        
        Args:
            question: Input question
            image: Document image (tensor or PIL Image)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer string
        """
        # Convert tensor to PIL Image if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]  # Remove batch dimension
            # Denormalize if needed
            if image.min() < 0:
                image = (image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + 
                        torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
            image = torch.clamp(image, 0, 1)
            # Convert to PIL
            image = Image.fromarray((image.permute(1, 2, 0) * 255).byte().cpu().numpy())
        
        # Create prompt based on model type
        model_type = self.get_model_type()
        
        if model_type == "llava":
            prompt = f"<image>\nUser: {question}\nAssistant:"
        elif model_type == "qwen":
            prompt = f"<|vision_start|><|image_pad|><|vision_end|>Question: {question}\nAnswer:"
        elif model_type == "phi":
            prompt = f"<|image_1|>\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}"
        
        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )
        
        # Decode answer
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (model-specific post-processing)
        if "Assistant:" in generated_text:
            answer = generated_text.split("Assistant:")[-1].strip()
        elif "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        return answer


class BaselineModelFactory:
    """Factory for creating baseline models."""
    
    @staticmethod
    def create_model(model_name: str, device: str = 'cuda') -> BaselineLLM:
        """
        Create baseline model based on model name.
        
        Args:
            model_name: Name/path of the model
            device: Device to load model on
            
        Returns:
            Baseline model instance
        """
        name_lower = model_name.lower()
        
        # Text-only models
        if any(text in name_lower for text in ["llama-2-7b-chat", "llama-3-8b-instruct"]) and "vision" not in name_lower:
            return TextOnlyBaseline(model_name, device)
        
        # Multi-modal models (text + bbox + image)
        elif any(mm in name_lower for mm in ["layoutlm", "doclay"]):
            return MultiModalBaseline(model_name, device)
        
        # Vision-language models (image only)
        elif any(vl in name_lower for vl in ["phi", "llava", "pixtral", "qwen", "internvl", "vision"]):
            return VisionLanguageBaseline(model_name, device)
        
        else:
            raise ValueError(f"Unknown model type: {model_name}")


class BaselineEvaluator:
    """Evaluator for comparing baseline models with MGA-VQA."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.load_baseline_models()
    
    def load_baseline_models(self):
        """Load all baseline models."""
        print("Loading baseline models...")
        
        # Text-only models
        for model_name in self.config.baselines.text_only_models:
            try:
                print(f"Loading {model_name}...")
                self.models[model_name] = BaselineModelFactory.create_model(model_name, self.config.device)
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
        
        # Multi-modal models
        for model_name in self.config.baselines.multimodal_models:
            try:
                print(f"Loading {model_name}...")
                self.models[model_name] = BaselineModelFactory.create_model(model_name, self.config.device)
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
        
        # Vision-language models
        for model_name in self.config.baselines.vision_language_models:
            try:
                print(f"Loading {model_name}...")
                self.models[model_name] = BaselineModelFactory.create_model(model_name, self.config.device)
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
        
        print(f"Successfully loaded {len(self.models)} baseline models")
    
    def evaluate_all_models(self, test_data: List[Dict]) -> Dict[str, Dict]:
        """
        Evaluate all baseline models on test data.
        
        Args:
            test_data: List of test samples with image, question, ocr_results, answer
            
        Returns:
            Dictionary of evaluation results for each model
        """
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            model_results = self.evaluate_single_model(model, test_data)
            results[model_name] = model_results
        
        return results
    
    def evaluate_single_model(self, model: BaselineLLM, test_data: List[Dict]) -> Dict:
        """Evaluate a single baseline model."""
        from mga_vqa.utils.metrics import compute_anls, MetricsTracker
        
        predictions = []
        targets = []
        
        for sample in test_data:
            question = sample['question']
            image = sample['image']
            ocr_results = sample.get('ocr_results', [])
            true_answer = sample['answer']
            
            # Generate prediction based on model type
            if isinstance(model, TextOnlyBaseline):
                ocr_texts = [item['text'] for item in ocr_results]
                pred_answer = model.generate_answer(question, ocr_texts)
            elif isinstance(model, MultiModalBaseline):
                pred_answer = model.generate_answer(question, image, ocr_results)
            elif isinstance(model, VisionLanguageBaseline):
                pred_answer = model.generate_answer(question, image)
            else:
                pred_answer = "Error"
            
            predictions.append(pred_answer)
            targets.append(true_answer)
        
        # Compute metrics
        # Convert text to token sequences for ANLS computation
        pred_tokens = [[i for i, c in enumerate(pred)] for pred in predictions]
        target_tokens = [[i for i, c in enumerate(target)] for target in targets]
        
        anls_score = compute_anls(pred_tokens, target_tokens)
        
        return {
            'anls': anls_score,
            'predictions': predictions,
            'targets': targets
        }


def create_baseline_evaluator(config):
    """Factory function to create baseline evaluator."""
    return BaselineEvaluator(config)
