# 🚀 Ultimate Guide: Training Polyp Segmentation on Google Colab

This guide provides two ways to run the training: **The Easy Way (Notebook)** and **The Manual Way (Terminal)**.

---

## ⚡ Option 1: The Easy Way (Recommended)

We have created a ready-to-use notebook `Train_on_Colab.ipynb` that automates everything.

1. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com/)
2. **Upload the Notebook**:
    * Click **Upload** -> Select `Train_on_Colab.ipynb` from this repository.
    * *OR* Select the **GitHub** tab -> Search for this repo -> Select `Train_on_Colab.ipynb`.
3. **Run All Cells**:
    * Go to **Runtime** > **Run all**.
    * It will automatically check your GPU, install libraries, download the dataset, and start the **Optimized Training**.

That's it! Just wait for the results.

---

## 🛠️ Option 2: The Manual Step-by-Step Way

If you prefer to run commands manually or want to understand the process, follow these steps.

### Step 1: Open Colab & Set GPU

1. Create a **New Notebook**.
2. **Runtime** > **Change runtime type** > Select **T4 GPU** (or V100/A100).
3. Click **Save**.

### Step 2: Clone the Repository

```python
!git clone https://github.com/kunal0230/Polyp-Segmentation-Using-UNet-3-with-TensorFlow.git
%cd Polyp-Segmentation-Using-UNet-3-with-TensorFlow
!git pull
```

### Step 3: Install Dependencies

```python
!pip install -q opencv-python scikit-learn pandas tensorflow
```

### Step 4: Download Dataset

```python
!wget https://datasets.simula.no/downloads/kvasir-seg.zip
!unzip -q kvasir-seg.zip
!mv Kvasir-SEG dataset
```

### Step 5: Start Optimized Training 🚀

Run this command to use **Mixed Precision** (2x speed) and **Cosine Decay** (better accuracy).

```python
!python train.py \
    --dataset_path "dataset" \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-3 \
    --img_size 256 \
    --use_cosine_lr \
    --mixed_precision
```

### Step 6: Monitor Progress

You can see the logs in real-time. If the session disconnects, just re-run the training command, and it will **automatically resume** from the last epoch!

### Step 7: Evaluate

```python
!python test.py --model_path "files/model.keras" --dataset_path "dataset" --img_size 256
```

---

## 🏆 Tips for Best Results

* **Always use T4 GPU or better.**
* **Mount Google Drive** to save your models permanently.
* **Use Mixed Precision**: It saves memory and doubles the speed.
* **Don't close the tab**: Colab will timeout if left idle. Use the "Keep Alive" script in `colab_setup.py` if needed.
