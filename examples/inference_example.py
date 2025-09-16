#!/usr/bin/env python3
"""
Example script for MGA-VQA inference.
Demonstrates how to use the trained model for document VQA predictions.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from PIL import Image
import json
import numpy as np
from mga_vqa.config import get_default_config
from mga_vqa.models.mga_vqa import MGA_VQA
from mga_vqa.training.trainer import MGAVQATrainer
from mga_vqa.utils.visualization import create_prediction_visualization
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MGA-VQA Inference')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to configuration file')
    
    # Input arguments
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input document image')
    parser.add_argument('--question', type=str, required=True,
                       help='Question about the document')
    parser.add_argument('--ocr_file', type=str, default=None,
                       help='Path to OCR results file (JSON)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/',
                       help='Directory to save outputs')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of results')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, config_file: str = None):
    """Load trained MGA-VQA model."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load configuration
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        # Convert to config object (simplified)
        config = get_default_config()
    else:
        config = get_default_config()
    
    # Load model from checkpoint
    if checkpoint_path.endswith('.ckpt'):
        # PyTorch Lightning checkpoint
        model = MGAVQATrainer.load_from_checkpoint(checkpoint_path).model
    else:
        # Regular PyTorch checkpoint
        model = MGA_VQA(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully!")
    return model, config


def load_image(image_path: str, target_size: tuple = (1024, 1024)):
    """Load and preprocess image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize while maintaining aspect ratio
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Convert to tensor
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_size, np.array(image)


def load_ocr_results(ocr_file: str = None, image_size: tuple = None):
    """Load OCR results from file or create dummy results."""
    if ocr_file and os.path.exists(ocr_file):
        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_results = json.load(f)
        print(f"Loaded {len(ocr_results)} OCR results from {ocr_file}")
    else:
        # Create dummy OCR results for demonstration
        print("No OCR file provided, creating dummy OCR results...")
        width, height = image_size if image_size else (1024, 768)
        
        ocr_results = [
            {
                'text': 'Invoice',
                'bbox': [0.1, 0.1, 0.3, 0.15]
            },
            {
                'text': 'Company Name: ABC Corp',
                'bbox': [0.1, 0.2, 0.5, 0.25]
            },
            {
                'text': 'Date: 2023-12-01',
                'bbox': [0.6, 0.2, 0.9, 0.25]
            },
            {
                'text': 'Total: $1,234.56',
                'bbox': [0.1, 0.8, 0.4, 0.85]
            },
            {
                'text': 'Customer: John Doe',
                'bbox': [0.1, 0.3, 0.5, 0.35]
            }
        ]
    
    return ocr_results


def run_inference(model, image_tensor, question, ocr_results, image_size, device):
    """Run inference on the model."""
    print(f"\nRunning inference...")
    print(f"Question: {question}")
    
    # Move to device
    image_tensor = image_tensor.to(device)
    model = model.to(device)
    
    # Prepare inputs
    questions = [question]
    ocr_results_batch = [ocr_results]
    image_sizes = [image_size]
    
    # Run inference
    with torch.no_grad():
        predictions = model.predict(
            image=image_tensor,
            question_text=questions,
            ocr_results=ocr_results_batch,
            image_sizes=image_sizes
        )
    
    # Extract results
    answer_sequence = predictions['answer_sequences'][0]
    answer_confidence = predictions['answer_confidence'][0]
    bbox_prediction = predictions['bbox_predictions'][0].cpu().numpy()
    
    # Convert answer sequence to text (simplified)
    # In a real implementation, you would use the vocabulary
    answer_text = f"answer_tokens_{len(answer_sequence)}"  # Placeholder
    
    results = {
        'question': question,
        'predicted_answer': answer_text,
        'answer_confidence': float(answer_confidence),
        'predicted_bbox': bbox_prediction.tolist(),
        'answer_tokens': answer_sequence
    }
    
    print(f"Predicted Answer: {answer_text}")
    print(f"Answer Confidence: {answer_confidence:.3f}")
    print(f"Predicted BBox: {bbox_prediction}")
    
    return results


def save_results(results, output_dir, base_name):
    """Save inference results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON results
    json_path = os.path.join(output_dir, f"{base_name}_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {json_path}")
    
    return json_path


def create_visualization(image_np, question, results, ocr_results, 
                        output_dir, base_name):
    """Create and save visualization."""
    print("Creating visualization...")
    
    # Extract results
    predicted_answer = results['predicted_answer']
    predicted_bbox = results['predicted_bbox']
    
    # Create visualization
    fig = create_prediction_visualization(
        image=image_np,
        question=question,
        predicted_answer=predicted_answer,
        true_answer="N/A",  # No ground truth in inference
        predicted_bbox=predicted_bbox,
        true_bbox=None,  # No ground truth
        ocr_results=ocr_results
    )
    
    # Save visualization
    vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    fig.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {vis_path}")
    
    return vis_path


def main():
    """Main function."""
    args = parse_args()
    
    print("MGA-VQA Inference Script")
    print("=" * 50)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, args.config_file)
    model = model.to(device)
    
    # Load image
    image_tensor, original_size, image_np = load_image(
        args.image, 
        target_size=(config.visual_encoder.max_image_size, 
                    config.visual_encoder.max_image_size)
    )
    print(f"Image loaded: {args.image} (size: {original_size})")
    
    # Load OCR results
    ocr_results = load_ocr_results(args.ocr_file, original_size)
    
    # Run inference
    results = run_inference(
        model, image_tensor, args.question, 
        ocr_results, original_size, device
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate base name for outputs
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    # Save results
    save_results(results, args.output_dir, base_name)
    
    # Create visualization if requested
    if args.visualize:
        create_visualization(
            image_np, args.question, results, 
            ocr_results, args.output_dir, base_name
        )
    
    print("\nInference completed successfully!")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
