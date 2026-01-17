# RGB to IR Image Translation Model

A physics-informed diffusion model for translating RGB images to corresponding Infrared (IR) images using:
- **SDXL Image-to-Image** as the base model
- **LoRA/LoHa** adaptation for parameter efficiency
- **ControlNet** for structure preservation (Canny edges, depth maps)
- **Physics-informed losses** (HADAR, thermal emissivity, transmitivity)

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│          RGB Input Image                        │
└──────────────────┬──────────────────────────────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
┌───▼────────────────┐   ┌───────▼────────────┐
│  Text Encoder      │   │   ControlNet       │
│  (Prompt)          │   │  - Canny Edge      │
│                    │   │  - Depth Map       │
└──────┬─────────────┘   └──────────┬─────────┘
       │                            │
       │  ┌────────────────────────┐│
       │  │                        ││
       ▼  ▼                        ▼│
┌──────────────────────────────────────┐
│    SDXL UNet + LoHA Adapters         │
│  ┌────────────────────────────────┐  │
│  │ LoHA Layers:                   │  │
│  │ - to_k: Material Recognition   │  │
│  │ - to_v: Emissivity/Transmitity │  │
│  │ - Self-Attention: Structure    │  │
│  │ - FF: Thermal Properties       │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
       │
       │ Physics-Informed Losses
       │ - HADAR Loss (thermal dynamics)
       │ - Perceptual Loss
       │ - Emissivity Loss
       │ - Transmitivity Loss
       │
       ▼
┌──────────────────────┐
│   IR Output Image    │
└──────────────────────┘
```

## Key Features

### 1. LoHA Adaptation Strategy
- **Rank**: 8-16 for parameter efficiency
- **Target Modules**:
  - `q_proj`, `k_proj`, `v_proj`: Attention mechanisms
  - `to_k`, `to_v`: Specialized for material and thermal properties
  - `out_proj`: Feature fusion
  - `ff`: Thermal dynamics

### 2. Physics-Informed Losses
- **HADAR Loss**: Thermal consistency from PID project
- **Perceptual Loss**: VQ-based perceptual features
- **Emissivity Loss**: Material surface property prediction
- **Transmitivity Loss**: Atmospheric transmission modeling
- **L1/L2 Losses**: Direct image reconstruction

### 3. ControlNet Integration
- **Canny Edge Detection**: Preserves structural edges
- **Depth Maps**: Maintains spatial relationships
- Both conditions weighted in denoising process

### 4. Dataset Format
```
RGB2IR_Dataset/
├── train/
│   ├── rgb/
│   │   ├── scene_001.png
│   │   └── ...
│   ├── ir/
│   │   ├── scene_001.png
│   │   └── ...
│   └── depth/  (optional)
│       ├── scene_001.npy
│       └── ...
├── val/
│   └── (same structure)
└── metadata.json
```

## Training

```bash
python train.py \
  --config configs/rgb2ir_loha.yaml \
  --data_path /path/to/dataset \
  --output_dir ./experiments/rgb2ir_v1 \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --epochs 100
```

## Inference

```bash
python inference.py \
  --checkpoint ./experiments/rgb2ir_v1/model.pt \
  --rgb_image input_rgb.png \
  --output output_ir.png \
  --use_controlnet \
  --controlnet_type depth
```

## Evaluation

```bash
python eval.py \
  --checkpoint ./experiments/rgb2ir_v1/model.pt \
  --val_dataset /path/to/validation \
  --metrics ssim,psnr,fid,thermal_consistency
```

## References

- SDXL: [Stability AI](https://huggingface.co/stabilityai/stable-diffusion-xl-1-0)
- LoHa: [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)
- ControlNet: [Zhang et al.](https://arxiv.org/abs/2302.05543)
- Physics-Informed Losses: PID Project
