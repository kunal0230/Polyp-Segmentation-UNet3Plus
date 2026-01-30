# UNet 3+ for Polyp Segmentation

This repository implements the UNet 3+ architecture in TensorFlow 2.x for medical image segmentation, specifically targeting the breakdown of polyps in the Kvasir-SEG dataset.

## Technical Architecture

The model builds upon the standard UNet by utilizing **full-scale skip connections**. Unlike UNet++ which uses nested dense pathways, UNet 3+ directly aggregates feature maps from:

1. **Same-scale encoder** (classic skip connection)
2. **Smaller-scale encoders** (via max-pooling)
3. **Larger-scale encoders** (via bilinear upsampling)

This allows the decoder at each stage to access both low-level spatial details and high-level semantic features simultaneously.

**Implementation Details:**

- **Backbone**: ResNet50 (pre-trained on ImageNet) used as the encoder.
- **Deep Supervision**: Disabled in this specific implementation; the model optimizes a single output from the final decoder layer.
- **Loss Function**: Dice Loss.
- **Metrics**: Dice Coefficient, IoU, Recall, Precision.

## Pipeline Optimization

The training loop (`train.py`) is optimized for high-throughput on NVIDIA GPUs (T4, V100, A100).

- **Mixed Precision (FP16)**: Utilizes `tensorflow.keras.mixed_precision` to perform compute in float16 while keeping variables in float32. This typically yields a 2x speedup and 50% memory reduction.
- **XLA JIT Compilation**: The model compilation enables XLA (`jit_compile=True`) to fuse kernels and optimize the computation graph.
- **Data Loading**:
  - `tf.data` pipeline with non-deterministic mapping for speed.
  - `cache()` enabled to keep processed augmentation in memory/local storage.
  - `prefetch(AUTOTUNE)` to overlap data preprocessing with GPU training steps.
- **Augmentation**:
  - Random 90-degree rotations (`k=[0..3]`).
  - Horizontal and Vertical flips.
  - Random brightness/contrast adjustments.

## Setup & Usage

### Dependencies

```bash
pip install -r requirements.txt
```

### Training

To run training with the optimized configuration:

```bash
python train.py --mixed_precision --use_cosine_lr --batch_size 16 --epochs 100
```

**Parameters:**

| Flag | Function | Default |
| :--- | :--- | :--- |
| `--dataset_path` | Directory containing `images/` and `masks/` | `Kvasir-SEG` |
| `--img_size` | Input resolution (square) | `256` |
| `--mixed_precision` | Enable mixed float16 training | `False` (Flag enables) |
| `--use_cosine_lr` | Use Cosine Decay with Linear Warmup | `False` (Flag enables) |

### Inference

Run evaluation on the test split:

```bash
python test.py --model_path "files/run_TIMESTAMP/model.keras" --save_path "results"
```

## Project Structure

```
.
├── train.py           # Main training entry point
├── test.py            # Evaluation and mask generation
├── model.py           # UNet 3+ architecture definition
├── metrics.py         # TensorFlow graph-compatible metrics
├── train_on_colab.ipynb # Quickstart notebook for Cloud/Colab usage
└── README.md          # Documentation
```

## References

- Huang, H., et al. "UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation." *ICASSP 2020*.
- Jha, D., et al. "Kvasir-SEG: A Segmented Polyp Dataset." *MultiMedia Modeling 2020*.
