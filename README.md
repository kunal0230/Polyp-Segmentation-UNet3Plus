# Polyp Segmentation Using UNet 3+ in TensorFlow

This project implements a deep learning model for segmenting polyps in medical images using the **UNet 3+** architecture in TensorFlow. It is designed to be easy to use, with support for Google Colab and command-line configuration.

## Features

- **UNet 3+ Architecture**: Advanced segmentation model with full-scale skip connections.
- **Metrics**: Dice Coefficient, IoU (Jaccard Index), Precision, Recall.
- **Data Augmentation**: Random flipping and rotation during training.
- **Configurable**: Easy-to-use command-line arguments for hyperparameters.
- **Colab Ready**: Includes `Train_on_Colab.ipynb` for one-click training on Google Colab.

## Dataset

The project uses the **Kvasir-SEG** dataset.

1. Download the dataset (e.g., from [Simula](https://datasets.simula.no/kvasir-seg/)).
2. Unzip it. You should have a folder `Kvasir-SEG` containing `images` and `masks`.

## Installation

### Local Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Polyp-Segmentation-using-UNet-3-Plus-in-TensorFlow-main
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Google Colab

1. Upload the `Kvasir-SEG.zip` and the project files (or clone the repo in Colab).
2. Open `Train_on_Colab.ipynb` and run the cells.

## Usage

### Training

Run `train.py` to train the model. You can specify parameters via command line:

```bash
python train.py --dataset_path "Kvasir-SEG" --epochs 100 --batch_size 4 --img_size 256
```

Arguments:

- `--dataset_path`: Path to the dataset folder (default: `Kvasir-SEG`).
- `--save_path`: Path to save model and logs (default: `files`).
- `--epochs`: Number of epochs (default: 100).
- `--batch_size`: Batch size (default: 2).
- `--lr`: Learning rate (default: 1e-4).
- `--img_size`: Image resolution (default: 256).

### Evaluation

Run `test.py` to evaluate the model and generate result images:

```bash
python test.py --model_path "files/model.keras" --dataset_path "Kvasir-SEG" --img_size 256
```

Arguments:

- `--model_path`: Path to the trained model file.
- `--save_path`: Path to save result images (default: `results`).
- `--img_size`: Image resolution (must match training resolution).

## Results

The `test.py` script will output quantitative metrics (Dice, IoU, Recall, Precision) and save visual comparisons in the `results/` folder.

Example output:

- **Dice Coefficient**: ~0.85
- **IoU**: ~0.75

## Model Architecture

UNet 3+ utilizes full-scale skip connections, combining high-level semantics with low-level details more effectively than standard UNet or UNet++.

## Credits

- **Paper**: UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation.
- **Dataset**: Kvasir-SEG.
