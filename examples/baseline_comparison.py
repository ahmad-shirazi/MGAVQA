#!/usr/bin/env python3
"""
Comprehensive baseline comparison script for MGA-VQA.
Compares MGA-VQA with all baseline LLMs mentioned in the research paper.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import json
import time
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mga_vqa.config import get_default_config
from mga_vqa.models.mga_vqa import MGA_VQA
from mga_vqa.models.baselines import BaselineEvaluator, BaselineModelFactory
from mga_vqa.utils.data_loader import create_synthetic_data_loader
from mga_vqa.utils.metrics import compute_anls, compute_iou_metrics, MetricsTracker
from mga_vqa.utils.visualization import plot_training_curves
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MGA-VQA Baseline Comparison')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/docvqa',
                       help='Directory containing dataset files')
    parser.add_argument('--use_synthetic', action='store_true',
                       help='Use synthetic data for testing')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of test samples to evaluate')
    
    # Model arguments
    parser.add_argument('--mga_checkpoint', type=str, default=None,
                       help='Path to trained MGA-VQA checkpoint')
    parser.add_argument('--baseline_models', nargs='+', default=None,
                       help='Specific baseline models to evaluate')
    
    # Evaluation arguments
    parser.add_argument('--max_new_tokens', type=int, default=50,
                       help='Maximum tokens to generate for answers')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='comparison_results/',
                       help='Directory to save comparison results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save detailed predictions for analysis')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    return parser.parse_args()


class ComprehensiveEvaluator:
    """Comprehensive evaluator for all models including MGA-VQA."""
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = self._setup_device()
        
        # Model categories from the paper
        self.text_only_models = [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Meta-Llama-3-8B-Instruct"
        ]
        
        self.multimodal_models = [
            "microsoft/LayoutLM-base-uncased",  # LayoutLLM-7B CoT equivalent
            "unstructured-io/Llama-2-7b-document",  # DocLayLLM (Llama2-7B) equivalent
            "unstructured-io/Llama-3-8b-document"   # DocLayLLM (Llama3-7B) equivalent
        ]
        
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
        
        # Initialize models
        self.mga_model = None
        self.baseline_models = {}
        self.results = {}
        
    def _setup_device(self):
        """Setup computation device."""
        if self.args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.args.device)
        
        print(f"Using device: {device}")
        return device
    
    def load_mga_vqa(self):
        """Load MGA-VQA model."""
        print("Loading MGA-VQA model...")
        
        if self.args.mga_checkpoint and os.path.exists(self.args.mga_checkpoint):
            # Load from checkpoint
            if self.args.mga_checkpoint.endswith('.ckpt'):
                from mga_vqa.training.trainer import MGAVQATrainer
                trainer = MGAVQATrainer.load_from_checkpoint(self.args.mga_checkpoint)
                self.mga_model = trainer.model
            else:
                self.mga_model = MGA_VQA(self.config)
                checkpoint = torch.load(self.args.mga_checkpoint, map_location=self.device)
                self.mga_model.load_state_dict(checkpoint.get('state_dict', checkpoint))
        else:
            # Create new model
            self.mga_model = MGA_VQA(self.config)
        
        self.mga_model = self.mga_model.to(self.device)
        self.mga_model.eval()
        print("✓ MGA-VQA model loaded successfully")
    
    def load_baseline_models(self):
        """Load selected baseline models."""
        print("Loading baseline models...")
        
        models_to_load = []
        if self.args.baseline_models:
            models_to_load = self.args.baseline_models
        else:
            # Load a subset for faster evaluation
            models_to_load = [
                self.text_only_models[0],  # Llama2-7B-Chat
                self.text_only_models[1],  # Llama3-8B-Instruct
                self.multimodal_models[0], # LayoutLLM equivalent
                self.vision_language_models[0],  # Phi4 equivalent
                self.vision_language_models[1],  # Llama3.2-11B-Vision
                self.vision_language_models[5],  # Qwen2-VL-7B
            ]
        
        for model_name in models_to_load:
            try:
                print(f"Loading {model_name}...")
                model = BaselineModelFactory.create_model(model_name, self.device)
                self.baseline_models[model_name] = model
                print(f"✓ {model_name} loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
        
        print(f"Successfully loaded {len(self.baseline_models)} baseline models")
    
    def create_test_data(self):
        """Create test dataset for evaluation."""
        if self.args.use_synthetic:
            print("Creating synthetic test data...")
            data_loader = create_synthetic_data_loader(
                batch_size=self.args.batch_size,
                num_samples=self.args.num_samples,
                num_workers=0
            )
            
            test_data = []
            for batch in data_loader:
                batch_size = batch['image'].shape[0]
                for i in range(batch_size):
                    sample = {
                        'image': batch['image'][i],
                        'question': batch['question'][i],
                        'ocr_results': batch['ocr_results'][i],
                        'image_size': batch['image_sizes'][i],
                        'answer': batch['answer_texts'][i],
                        'bbox': batch['bbox_targets'][i].tolist()
                    }
                    test_data.append(sample)
            
            print(f"Created {len(test_data)} test samples")
            return test_data[:self.args.num_samples]
        
        else:
            # TODO: Load real dataset
            print("Real dataset loading not implemented, falling back to synthetic data")
            self.args.use_synthetic = True
            return self.create_test_data()
    
    def evaluate_mga_vqa(self, test_data: List[Dict]) -> Dict:
        """Evaluate MGA-VQA model."""
        print("Evaluating MGA-VQA...")
        
        predictions = []
        targets = []
        bbox_predictions = []
        bbox_targets = []
        inference_times = []
        
        with torch.no_grad():
            for sample in test_data:
                start_time = time.time()
                
                # Prepare inputs
                image = sample['image'].unsqueeze(0).to(self.device)
                questions = [sample['question']]
                ocr_results = [sample['ocr_results']]
                image_sizes = [sample['image_size']]
                
                # Run inference
                pred_results = self.mga_model.predict(
                    image=image,
                    question_text=questions,
                    ocr_results=ocr_results,
                    image_sizes=image_sizes
                )
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Extract predictions
                pred_answer = pred_results['answer_sequences'][0]
                pred_bbox = pred_results['bbox_predictions'][0].cpu().tolist()
                
                predictions.append(pred_answer)
                targets.append(sample['answer'])
                bbox_predictions.append(pred_bbox)
                bbox_targets.append(sample['bbox'])
        
        # Compute metrics
        # Convert to appropriate format for ANLS
        pred_tokens = [[i for i, c in enumerate(str(pred))] for pred in predictions]
        target_tokens = [[i for i, c in enumerate(str(target))] for target in targets]
        
        anls_score = compute_anls(pred_tokens, target_tokens)
        
        # IoU metrics
        bbox_preds_tensor = torch.tensor(bbox_predictions)
        bbox_targets_tensor = torch.tensor(bbox_targets)
        iou_metrics = compute_iou_metrics(bbox_preds_tensor, bbox_targets_tensor)
        
        results = {
            'model_name': 'MGA-VQA (Gemma-3-8B)',
            'model_type': 'Multi-Modal Graph-Augmented',
            'anls': anls_score,
            'mean_iou': iou_metrics['val_mean_iou'],
            'iou_0.5': iou_metrics['val_iou_0.5'],
            'iou_0.75': iou_metrics['val_iou_0.75'],
            'map': iou_metrics['val_map'],
            'avg_inference_time': sum(inference_times) / len(inference_times),
            'predictions': predictions if self.args.save_predictions else [],
            'targets': targets if self.args.save_predictions else []
        }
        
        print(f"MGA-VQA Results: ANLS={anls_score:.3f}, IoU@0.5={iou_metrics['val_iou_0.5']:.3f}")
        return results
    
    def evaluate_baseline_model(self, model_name: str, model, test_data: List[Dict]) -> Dict:
        """Evaluate a single baseline model."""
        print(f"Evaluating {model_name}...")
        
        predictions = []
        targets = []
        inference_times = []
        
        for sample in test_data:
            start_time = time.time()
            
            try:
                # Generate prediction based on model type
                if any(text_model in model_name for text_model in self.text_only_models):
                    ocr_texts = [item['text'] for item in sample['ocr_results']]
                    pred_answer = model.generate_answer(
                        sample['question'], ocr_texts, self.args.max_new_tokens
                    )
                elif any(mm_model in model_name for mm_model in self.multimodal_models):
                    pred_answer = model.generate_answer(
                        sample['question'], sample['image'], sample['ocr_results'], 
                        self.args.max_new_tokens
                    )
                else:  # Vision-language models
                    pred_answer = model.generate_answer(
                        sample['question'], sample['image'], self.args.max_new_tokens
                    )
                
            except Exception as e:
                print(f"Error in {model_name}: {e}")
                pred_answer = "Error"
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            predictions.append(pred_answer)
            targets.append(sample['answer'])
        
        # Compute ANLS metric
        pred_tokens = [[i for i, c in enumerate(str(pred))] for pred in predictions]
        target_tokens = [[i for i, c in enumerate(str(target))] for target in targets]
        
        anls_score = compute_anls(pred_tokens, target_tokens)
        
        # Determine model type
        if any(text_model in model_name for text_model in self.text_only_models):
            model_type = "Text Only"
        elif any(mm_model in model_name for mm_model in self.multimodal_models):
            model_type = "Text + BBox + Image"
        else:
            model_type = "Image Only"
        
        results = {
            'model_name': model_name.split('/')[-1],  # Clean model name
            'model_type': model_type,
            'anls': anls_score,
            'mean_iou': 0.0,  # Baselines don't predict bboxes
            'iou_0.5': 0.0,
            'iou_0.75': 0.0,
            'map': 0.0,
            'avg_inference_time': sum(inference_times) / len(inference_times),
            'predictions': predictions if self.args.save_predictions else [],
            'targets': targets if self.args.save_predictions else []
        }
        
        print(f"{model_name} Results: ANLS={anls_score:.3f}")
        return results
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation of all models."""
        print("=" * 60)
        print("MGA-VQA Comprehensive Baseline Comparison")
        print("=" * 60)
        
        # Create test data
        test_data = self.create_test_data()
        
        # Load models
        self.load_mga_vqa()
        self.load_baseline_models()
        
        # Evaluate MGA-VQA
        self.results['MGA-VQA'] = self.evaluate_mga_vqa(test_data)
        
        # Evaluate baseline models
        for model_name, model in self.baseline_models.items():
            self.results[model_name] = self.evaluate_baseline_model(
                model_name, model, test_data
            )
        
        return self.results
    
    def save_results(self, results: Dict):
        """Save evaluation results."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(self.args.output_dir, 'detailed_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary DataFrame
        summary_data = []
        for model_key, result in results.items():
            summary_data.append({
                'Model': result['model_name'],
                'Type': result['model_type'],
                'ANLS': result['anls'],
                'IoU@0.5': result['iou_0.5'],
                'IoU@0.75': result['iou_0.75'],
                'mAP': result['map'],
                'Inference Time (s)': result['avg_inference_time']
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('ANLS', ascending=False)
        
        # Save CSV
        df.to_csv(os.path.join(self.args.output_dir, 'summary_results.csv'), index=False)
        
        # Print summary table
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False, float_format='%.3f'))
        
        return df
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create comparison visualizations."""
        print("\nCreating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # ANLS comparison by model type
        ax1 = axes[0, 0]
        sns.barplot(data=df, x='Type', y='ANLS', ax=ax1, palette='viridis')
        ax1.set_title('ANLS Score by Model Type')
        ax1.set_ylabel('ANLS Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Individual model ANLS scores
        ax2 = axes[0, 1]
        df_sorted = df.sort_values('ANLS', ascending=True)
        bars = ax2.barh(df_sorted['Model'], df_sorted['ANLS'], 
                       color=['red' if 'MGA-VQA' in model else 'blue' 
                             for model in df_sorted['Model']])
        ax2.set_title('ANLS Scores by Model')
        ax2.set_xlabel('ANLS Score')
        
        # Highlight MGA-VQA
        for i, bar in enumerate(bars):
            if 'MGA-VQA' in df_sorted.iloc[i]['Model']:
                bar.set_color('red')
                bar.set_alpha(0.8)
        
        # IoU scores (only for models that predict bboxes)
        ax3 = axes[1, 0]
        iou_data = df[df['IoU@0.5'] > 0]
        if not iou_data.empty:
            ax3.bar(['IoU@0.5', 'IoU@0.75', 'mAP'], 
                   [iou_data['IoU@0.5'].iloc[0], iou_data['IoU@0.75'].iloc[0], iou_data['mAP'].iloc[0]],
                   color='orange', alpha=0.7)
            ax3.set_title('MGA-VQA Localization Performance')
            ax3.set_ylabel('Score')
        else:
            ax3.text(0.5, 0.5, 'No Localization Results', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Localization Performance')
        
        # Inference time comparison
        ax4 = axes[1, 1]
        ax4.scatter(df['ANLS'], df['Inference Time (s)'], 
                   s=100, alpha=0.7, c=df.index, cmap='tab10')
        for i, model in enumerate(df['Model']):
            ax4.annotate(model[:15] + '...' if len(model) > 15 else model, 
                        (df.iloc[i]['ANLS'], df.iloc[i]['Inference Time (s)']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('ANLS Score')
        ax4.set_ylabel('Inference Time (seconds)')
        ax4.set_title('Performance vs Speed Trade-off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'comparison_plots.png'), 
                   dpi=300, bbox_inches='tight')
        
        # Create detailed comparison table plot
        fig2, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap of normalized scores
        metrics = ['ANLS', 'IoU@0.5', 'IoU@0.75', 'mAP']
        heatmap_data = df[metrics].copy()
        
        # Normalize each column to 0-1 for better visualization
        for col in metrics:
            max_val = heatmap_data[col].max()
            if max_val > 0:
                heatmap_data[col] = heatmap_data[col] / max_val
        
        sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', 
                   xticklabels=df['Model'], yticklabels=metrics,
                   cmap='RdYlGn', ax=ax)
        ax.set_title('Normalized Performance Metrics Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'performance_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        
        print(f"Visualizations saved to {self.args.output_dir}")


def main():
    """Main function."""
    args = parse_args()
    
    # Setup configuration
    config = get_default_config()
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(config, args)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Save and visualize results
    df = evaluator.save_results(results)
    evaluator.create_visualizations(df)
    
    print(f"\nEvaluation completed! Results saved to {args.output_dir}")
    
    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    mga_result = results.get('MGA-VQA', {})
    if mga_result:
        print(f"🏆 MGA-VQA (Gemma-3-8B) Performance:")
        print(f"   • ANLS Score: {mga_result['anls']:.3f}")
        print(f"   • IoU@0.5: {mga_result['iou_0.5']:.3f}")
        print(f"   • Mean IoU: {mga_result['mean_iou']:.3f}")
        print(f"   • Inference Time: {mga_result['avg_inference_time']:.3f}s")
    
    best_baseline = df.iloc[1] if len(df) > 1 else None  # Second best (first is MGA-VQA)
    if best_baseline is not None:
        print(f"\n🥈 Best Baseline Model: {best_baseline['Model']}")
        print(f"   • ANLS Score: {best_baseline['ANLS']:.3f}")
        print(f"   • Model Type: {best_baseline['Type']}")
        print(f"   • Inference Time: {best_baseline['Inference Time (s)']:.3f}s")
    
    if mga_result and best_baseline is not None:
        improvement = (mga_result['anls'] - best_baseline['ANLS']) / best_baseline['ANLS'] * 100
        print(f"\n📈 MGA-VQA Improvement: +{improvement:.1f}% ANLS over best baseline")


if __name__ == '__main__':
    main()
