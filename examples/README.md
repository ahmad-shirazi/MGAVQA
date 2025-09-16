# MGA-VQA Examples

This directory contains example scripts demonstrating how to use the MGA-VQA framework with Gemma-3-8B backbone.

## 📚 Available Examples

### 1. Training Script (`train_mga_vqa.py`)
Complete training pipeline with multi-stage strategy support.

```bash
# Quick test with synthetic data
python train_mga_vqa.py --use_synthetic --stage 4 --batch_size 8 --max_epochs 2

# Multi-stage training with real data
python train_mga_vqa.py --data_dir /path/to/docvqa --multi_stage --logger wandb

# Single stage training with Gemma backbone
python train_mga_vqa.py --data_dir /path/to/docvqa --stage 4 --batch_size 16
```

**Features:**
- Multi-stage training (4 stages as described in paper)
- Gemma-3-8B integration
- PyTorch Lightning support
- WandB/TensorBoard logging
- Synthetic data generation for testing

### 2. Inference Script (`inference_example.py`)
End-to-end inference with visualization capabilities.

```bash
python inference_example.py \
    --checkpoint checkpoints/mga_vqa_final.ckpt \
    --image /path/to/document.jpg \
    --question "What is the invoice number?" \
    --ocr_file /path/to/ocr_results.json \
    --visualize
```

**Features:**
- Single image inference
- OCR integration support
- Answer prediction and localization
- Visualization generation
- Gemma-enhanced answer generation

### 3. Baseline Comparison (`baseline_comparison.py`)
Comprehensive comparison with 13+ state-of-the-art LLMs.

```bash
# Full comparison with all baseline models
python baseline_comparison.py \
    --use_synthetic \
    --num_samples 500 \
    --output_dir comparison_results/ \
    --save_predictions

# Compare with specific models only
python baseline_comparison.py \
    --baseline_models "meta-llama/Llama-2-7b-chat-hf" "Qwen/Qwen2-VL-7B-Instruct" \
    --mga_checkpoint checkpoints/mga_vqa_final.ckpt
```

**Features:**
- Evaluation against all paper baselines
- Performance metrics (ANLS, IoU, mAP)
- Statistical analysis and visualizations
- Inference time comparisons
- Detailed results export

**Supported Baseline Categories:**

**Text-Only Models:**
- Llama2-7B-Chat
- Llama3-8B-Instruct

**Text + BBox + Image Models:**
- LayoutLLM-7B CoT (Vicuna backbone)
- DocLayLLM (Llama2-7B)
- DocLayLLM (Llama3-7B)

**Image-Only Models:**
- Phi4-14B
- Llama3.2-11B-Vision
- Pixtral-12B
- LLaVA-NeXT-13B
- LLaVA-OneVision-7B
- Qwen2.5-VL-7B
- InternVL2-8B
- DLaVA (Pixtral-12B)

### 4. Component Testing (`component_test.py`)
Comprehensive testing suite for all MGA-VQA components.

```bash
python component_test.py
```

**Tests Include:**
- ✓ Token-Level Visual Encoder
- ✓ Spatial Graph Construction & Reasoning
- ✓ Memory-Augmented Question Processing (with Gemma)
- ✓ Question-Guided Compression
- ✓ Multi-Modal Spatial Fusion
- ✓ Complete Model Pipeline
- ✓ Data Loading Utilities
- ✓ Visualization Functions

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# For baseline comparisons, you may need additional model-specific packages
pip install flash-attn accelerate bitsandbytes
```

### Run Your First Experiment

1. **Test the Complete Pipeline:**
```bash
python component_test.py
```

2. **Train on Synthetic Data:**
```bash
python train_mga_vqa.py --use_synthetic --max_epochs 2
```

3. **Run Baseline Comparison:**
```bash
python baseline_comparison.py --use_synthetic --num_samples 100
```

## 📊 Expected Outputs

### Training Results
- Model checkpoints saved in `checkpoints/`
- Training logs in `logs/`
- Performance metrics tracked via WandB/TensorBoard

### Inference Results
- Generated answers with confidence scores
- Predicted bounding boxes for answer localization
- Visualization images showing attention and predictions

### Baseline Comparison Results
- **Summary CSV**: `comparison_results/summary_results.csv`
- **Detailed JSON**: `comparison_results/detailed_results.json`
- **Visualizations**: 
  - `comparison_plots.png`: Performance comparisons
  - `performance_heatmap.png`: Normalized metrics heatmap

## 🔧 Customization

### Configuration
Modify `mga_vqa/config.py` to customize:
- **Backbone Model**: Change `backbone_model` parameter
- **Architecture**: Adjust layer dimensions and counts
- **Training**: Modify learning rates, batch sizes, etc.

### Adding New Baselines
1. Implement model in `mga_vqa/models/baselines.py`
2. Add model name to appropriate category in config
3. Test with comparison script

### Custom Datasets
1. Implement dataset class following `DocumentVQADataset` pattern
2. Update data loader creation in training script
3. Ensure proper OCR result formatting

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory:**
- Reduce batch size: `--batch_size 4`
- Use gradient accumulation: Update config
- Enable mixed precision: `--precision 16-mixed`

**2. Model Loading Errors:**
- Check HuggingFace model availability
- Ensure proper authentication for gated models
- Verify model name spelling

**3. Baseline Model Issues:**
- Some models may not be publicly available
- Use `--baseline_models` to select specific models
- Check model compatibility with your GPU

### Getting Help

- 📖 Check the main README for detailed documentation
- 🐛 Open an issue on GitHub for bugs
- 💬 Join our Discord for community support
- 📧 Contact the team for enterprise support

## 📈 Performance Tips

1. **Use Synthetic Data** for quick testing and development
2. **Enable Mixed Precision** for faster training on modern GPUs
3. **Use Gradient Checkpointing** for larger batch sizes
4. **Monitor GPU Memory** usage with `nvidia-smi`
5. **Profile Your Code** to identify bottlenecks

---

**Happy experimenting with MGA-VQA! 🎉**
