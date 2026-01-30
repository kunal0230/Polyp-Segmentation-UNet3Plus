# UNet 3+ Polyp Segmentation

TensorFlow implementation of UNet 3+ trained on the Kvasir-SEG dataset.

## Architecture

The model implements UNet 3+ (Huang et al., 2020), which uses full-scale skip connections to combine low-level details with high-level semantics from feature maps at different scales.

Validation metrics:

- Dice Coefficient
- IoU
- Precision/Recall

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Training**

```bash
# Default training (T4/V100/A100 optimized)
python train.py --mixed_precision --use_cosine_lr --batch_size 16

# CPU/Low-memory GPU
python train.py --batch_size 8
```

**Inference**

```bash
python test.py --model_path files/run_<timestamp>/model.keras
```

## Colab

A fast-track notebook `train_on_colab.ipynb` is included for cloud training.
It handles the Kvasir-SEG download loop and environment constraints (T4 GPU drivers, etc).
