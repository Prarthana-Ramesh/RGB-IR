# Quick Start Guide - RGB to IR Image Translation Model

## Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install xformers for memory efficiency
pip install xformers
```

### 2. Download Pretrained Models

The model automatically downloads SDXL and ControlNet models from Hugging Face on first run:
- SDXL: `stabilityai/stable-diffusion-xl-1-0`
- ControlNet Depth: `lllyasviel/control_v11f1p_sd15_depth`
- ControlNet Canny: `lllyasviel/control_v11p_sd15_canny`

You need ~50GB of disk space for all models.

### 3. Prepare Your Dataset

Organize your aligned RGB-IR image pairs:

```
data/RGB2IR_dataset/
├── train/
│   ├── rgb/
│   │   ├── scene_001.png
│   │   ├── scene_002.png
│   │   └── ...
│   ├── ir/
│   │   ├── scene_001.png
│   │   ├── scene_002.png
│   │   └── ...
│   └── depth/  (optional - auto-estimated if missing)
│       ├── scene_001.npy
│       └── ...
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

**Notes:**
- RGB images: standard 3-channel PNG/JPG
- IR images: single-channel thermal images (8-bit or 16-bit)
- Depth maps: can be `.npy` files or single-channel PNG images
- If depth maps are not provided, they're automatically estimated from RGB edges
- Canny edge maps are computed automatically from RGB

### 4. Update Configuration

Edit `configs/train_config.yaml`:

```yaml
data:
  dataset_root: "/path/to/RGB2IR_dataset"
  image_size: [512, 512]  # Your image size

training:
  batch_size: 4
  epochs: 100
  output_dir: "./experiments/rgb2ir_v1"
```

## Training

### Basic Training

```bash
python train.py --config configs/train_config.yaml --device cuda
```

### Resume Training

```bash
python train.py --config configs/train_config.yaml --resume experiments/rgb2ir_v1/checkpoints/best.pt --device cuda
```

### Distributed Training (Multi-GPU)

```bash
torchrun --nproc_per_node=4 train.py --config configs/train_config.yaml
```

## Inference

### Single Image Translation

```bash
python inference.py \
  --config configs/rgb2ir_loha.yaml \
  --checkpoint experiments/rgb2ir_v1/checkpoints/best.pt \
  --rgb_image test_rgb.png \
  --output output_ir.png \
  --use_controlnet \
  --controlnet_type both \
  --colormap
```

### Batch Processing

```bash
for rgb_file in rgb_images/*.png; do
  python inference.py \
    --config configs/rgb2ir_loha.yaml \
    --checkpoint experiments/rgb2ir_v1/checkpoints/best.pt \
    --rgb_image "$rgb_file" \
    --output "ir_outputs/$(basename $rgb_file)" \
    --use_controlnet
done
```

## Evaluation

### Full Dataset Evaluation

```bash
python eval.py \
  --config configs/rgb2ir_loha.yaml \
  --checkpoint experiments/rgb2ir_v1/checkpoints/best.pt \
  --dataset data/RGB2IR_dataset \
  --split val \
  --batch_size 1
```

Metrics computed:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MAE/MSE**: Pixel-level reconstruction error
- **Gradient Matching**: Edge preservation quality
- **Thermal Consistency**: Temperature smoothness

## Model Architecture Details

### LoHA Adaptation Strategy

The model uses **Low-rank Hadamard Product (LoHA)** for efficient fine-tuning:

- **Text Encoder**: Rank 16 (semantic understanding)
- **UNet**: Rank 8 (parameter efficiency)
- **Specialized Modules**:
  - `to_k` projections: Learn **material recognition** (reflectance, emissivity)
  - `to_v` projections: Learn **thermal properties** (temperature, emissivity)
  - Self-attention: **Structure preservation** (geometric consistency)

### Physics-Informed Losses

The model incorporates multiple loss components:

1. **L1 Loss**: Direct pixel-level reconstruction
2. **HADAR Loss**: Thermal dynamics consistency
   - Gradient matching (temperature transitions)
   - Smoothness enforcement (Laplacian regularization)
3. **Emissivity Loss**: Material property prediction
   - Learns from to_k projections
   - Encodes surface material characteristics
4. **Transmitivity Loss**: Atmospheric effects
   - Models long-range dependencies
   - Preserves spatial coherence
5. **Perceptual Loss**: Feature-level similarity
   - Based on VQ-VAE features
6. **Structure Loss**: Attention map consistency
   - Preserves geometric structure from ControlNet

### ControlNet Integration

Two control signals guide generation:

1. **Depth Map Control** (scale: 1.0)
   - Preserves spatial relationships
   - Auto-estimated from RGB if not provided

2. **Canny Edge Control** (scale: 0.7)
   - Preserves structural edges
   - Ensures sharp thermal boundaries

## Advanced Usage

### Custom Material Recognition

The model learns material properties through the `to_k` projections. You can extract material embeddings:

```python
from models.rgb2ir_model import RGB2IRLoHaModel

model = RGB2IRLoHaModel(config_path='configs/rgb2ir_loha.yaml')
# Material embeddings available in attention keys during forward pass
```

### Thermal Property Analysis

Extract predicted thermal properties:

```python
import torch
material_logits, reflectance = model.material_recognition(key_features)
emissivity, temp_offset = model.emissivity_calculation(value_features)
```

### Memory-Efficient Inference

For inference on limited GPU memory:

```python
model.enable_model_cpu_offload()  # Offload to CPU when not needed
model.enable_attention_slicing()   # Reduce attention memory
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size in config
batch_size: 2

# Or use smaller image size
image_size: [256, 256]

# Or enable memory-efficient attention
# In inference.py, call: model.enable_attention_slicing()
```

### Slow Training

```bash
# Reduce inference steps during training
num_inference_steps: 20  # In train.py

# Disable ControlNet if not needed
enable_controlnet: false

# Use mixed precision (enabled by default with float16)
```

### Poor Quality Results

1. **Check data alignment**: Ensure RGB and IR images are perfectly aligned
2. **Verify normalization**: Check if IR image normalization parameters match your data
3. **More training**: Increase epochs and use validation metrics to monitor progress
4. **Adjust guidance scale**: Try different values (5.0-10.0) for better control

## Citation & References

```bibtex
@article{podell2023sdxl,
  title={SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis},
  author={Podell, David and others},
  year={2023}
}

@article{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Zhang, Lvmin and others},
  journal={arXiv preprint arXiv:2302.05543},
  year={2023}
}

@article{kuboyama2024loha,
  title={LoRA-HA: Low-Rank Hadamard-Product Adaptation},
  author={Kuboyama, Koji and others},
  journal={arXiv preprint},
  year={2024}
}
```

## Support

For issues and questions:
1. Check the [Diffusers documentation](https://huggingface.co/docs/diffusers/)
2. Review SDXL model card and ControlNet documentation
3. Check PID project documentation for physics-informed losses
4. Review LoHA/PEFT documentation for adapter details

---

**Last Updated**: January 2026
**Model Version**: RGB2IR-LoHA v1.0
