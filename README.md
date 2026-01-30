# Polyp Segmentation using UNet 3+ in TensorFlow

This repository contains an optimized TensorFlow implementation of **UNet 3+** for automatic polyp segmentation in medical imaging, specifically designed for the **Kvasir-SEG** dataset.

## Key Features

* **UNet 3+ Architecture**: Advanced architecture ensuring full-scale skip connections for better feature aggregation.
* **Optimized Data Pipeline**:
  * **Mixed Precision Training**: Levarages Float16 operations for faster training on Tensor Core GPUs (T4, V100, A100).
  * **Native TensorFlow I/O**: Efficient image decoding and resizing avoiding CPU bottlenecks.
  * **Caching & Prefetching**: Maximizes GPU utilization by loading data ahead of time.
* **Advanced Training Techniques**:
  * **Cosine Annealing Schedule**: Smooth learning rate decay for better convergence.
  * **Data Augmentation**: Robust pipeline including rotations, flips, brightness, and contrast adjustments.
  * **XLA Compilation**: Uses Just-In-Time (JIT) compilation for graph optimization.
* **Robustness**:
  * **Automatic Resuming**: Automatically detects and resumes training from the last saved epoch in case of interruptions.
  * **Memory Management**: Custom callbacks to prevent OOM errors during long training sessions.

## Dataset

This project uses the **Kvasir-SEG** dataset, which consists of 1,000 polyp images and their corresponding ground truth masks.

* Download from: [Simula Datasets](https://datasets.simula.no/kvasir-seg/)
* The script handles automatic downloading and extraction when running on Colab.

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**

* TensorFlow 2.x
* OpenCV
* Pandas
* Scikit-learn

## Usage

### 1. Training

To train the model with default settings (optimized for 16GB VRAM GPUs):

```bash
python train.py --epochs 100 --batch_size 16 --lr 1e-3
```

**Advanced Options:**

```bash
python train.py \
    --dataset_path "Kvasir-SEG" \
    --img_size 256 \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-3 \
    --use_cosine_lr \
    --mixed_precision
```

### 2. Google Colab

For users without local GPUs, we provide a streamlined Colab notebook.

* Open `train_on_colab.ipynb` in Google Colab.
* Run the cells to check GPU, clone the repository, and start training.

### 3. Evaluation

To evaluate the trained model on the test set:

```bash
python test.py --model_path "files/model.keras" --dataset_path "Kvasir-SEG"
```

Results (images and masks) will be saved in the `results/` directory.

## File Structure

* `train.py`: Main training script with optimization logic.
* `test.py`: Evaluation script for generating predictions.
* `model.py`: UNet 3+ architecture definition.
* `metrics.py`: Custom metrics (Dice, IoU) compliant with TensorFlow graph execution.
* `train_on_colab.ipynb`: One-click Jupyter notebook for Colab training.

## References

* *UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation* (Huang et al., 2020)
* *Kvasir-SEG: A Segmented Polyp Dataset* (Jha et al., 2020)
