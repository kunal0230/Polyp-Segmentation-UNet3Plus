# Step-by-Step Guide: Training Polyp Segmentation on Google Colab

This guide will walk you through setting up and running the training process on Google Colab, even if you are new to it.

## Prerequisites

1. **Google Account**: To access Google Colab and Drive.
2. **Dataset**: The `Kvasir-SEG` dataset downloaded on your local computer as a `.zip` file (e.g., `Kvasir-SEG.zip`).
    * Inside the zip, it should have `images/` and `masks/` folders.
3. **GitHub Account**: To access the code repository.

## Step 1: Open Google Colab and Set GPU

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **"New Notebook"** in the bottom right, or select **GitHub** tab.
3. **Action**: In the top menu, go to **Runtime** > **Change runtime type**.
4. **Action**: Under **Hardware accelerator**, select **T4 GPU** (or any available GPU).
5. Click **Save**. This is crucial for fast training!

## Step 2: Clone the Repository

In the first code cell of your notebook, run the following command to download your code into the Colab environment.

```python
!git clone https://github.com/kunal0230/Polyp-Segmentation-UNet3Plus.git
%cd Polyp-Segmentation-UNet3Plus
```

* `!git clone` downloads the code.
* `%cd` changes the directory so you are inside the project folder.

## Step 3: Install Dependencies

Run the next cell to install the required libraries.

```python
!pip install -r requirements.txt
```

## Step 4: Upload the Dataset

You have two options here.

### Option A: Upload directly (Easiest for small files)

1. Click the **Folder icon** on the left sidebar (Files).
2. Click the **Upload icon** (file with an arrow).
3. Select your `Kvasir-SEG.zip` file from your computer.
4. Wait for the upload to finish (can take time depending on internet speed).

### Option B: Mount Google Drive (Faster if dataset is already on Drive)

1. Upload `Kvasir-SEG.zip` to your Google Drive first.
2. In Colab, run:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3. Copy the zip file to the current folder:

    ```python
    !cp /content/drive/MyDrive/path/to/Kvasir-SEG.zip .
    ```

## Step 5: Unzip the Dataset

Run this command to unzip the dataset into the project folder.

```python
!unzip -q Kvasir-SEG.zip
```

* You should now see a `Kvasir-SEG` folder in the Files sidebar.

## Step 6: Start Training

Now you are ready to train! The script has been optimized for Colab (Mixed Precision, XLA, Caching).

 **Basic Command:**

 ```python
 !python train.py --dataset_path "Kvasir-SEG" --epochs 50 --batch_size 16 --lr 1e-3 --img_size 256
 ```

 **Optimized Command (Recommended):**

 ```python
 !python train.py \
     --dataset_path "Kvasir-SEG" \
     --epochs 100 \
     --batch_size 16 \
     --lr 1e-3 \
     --img_size 256 \
     --use_cosine_lr \
     --mixed_precision
 ```

 **Parameters:**

* `--use_cosine_lr`: Enables Cosine Annealing scheduler (better results).
* `--mixed_precision`: Enables FP16 training (2x faster).
* `--batch_size`: 16 is good for T4 GPU. Use 8 if OOM.
* `--img_size`: 256 is standard. 320 gives better accuracy but is slower.

## Step 7: Monitor Training

You will see output like this for each epoch:

```
Epoch 1/50
100/100 [==============================] - 10s 100ms/step - loss: 0.5 - dice_coef: 0.5 - val_loss: 0.4 ...
```

Wait for it to finish. The best model will be saved automatically to `files/model.keras`.

## Resuming Training

The training script `train.py` now supports **automatic resuming**.
If your Colab runtime disconnects:

1. **Mount Drive** (if you saved files there).
2. **Pull latest code**: `!git pull`
3. **Run training again**: `!python train.py ...`

The script will detect the existing `model.keras` and `log.csv` and resume from the last saved epoch automatically.

## Step 8: Evaluate and Test

To see how well the model works on unseen data, run:

```python
!python test.py --model_path "files/model.keras" --dataset_path "Kvasir-SEG" --img_size 256
```

* This will generate predicted images in the `results/` folder.

## Step 9: Visualize and Download Results

To quickly see a result in the notebook:

```python
import cv2
from google.colab.patches import cv2_imshow
import glob

# specific result or random one
clean_image = cv2.imread(glob.glob("results/*.jpg")[0])
cv2_imshow(clean_image)
```

To download all results and the trained model to your computer:

```python
!zip -r training_output.zip files results
from google.colab import files
files.download('training_output.zip')
```

**That's it! You have successfully trained and evaluated a deep learning model for medical image segmentation.**
