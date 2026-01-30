import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Enable memory growth for GPUs to prevent OOM errors
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import pandas as pd
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
from model import unet3plus
from metrics import dice_loss, dice_coef, iou
import argparse
from datetime import datetime
import gc

# Enable Mixed Precision Training
def setup_gpu():
    """Configure GPU settings for optimal performance on Colab"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print(f'Mixed precision enabled: {policy.name}')
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found. Training will be slow on CPU.")

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.1):
    X = sorted(glob(os.path.join(path, "images", "*")))
    Y = sorted(glob(os.path.join(path, "masks", "*")))
    
    if len(X) == 0 or len(Y) == 0:
        raise ValueError(f"No images or masks found in {path}")
    
    if len(X) != len(Y):
        raise ValueError(f"Mismatch: {len(X)} images but {len(Y)} masks")
    
    print(f"Total dataset size: {len(X)} images")

    split_size = max(1, int(len(X) * split))
    
    train_val_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_val_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)
    
    train_x, valid_x = train_test_split(train_val_x, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(train_val_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path, img_h, img_w):
    x = tf.io.read_file(path)
    x = tf.image.decode_image(x, channels=3, expand_animations=False)
    x = tf.image.resize(x, (img_h, img_w), method='bilinear')
    x = tf.cast(x, tf.float32) / 255.0
    x.set_shape([img_h, img_w, 3])
    return x

def read_mask(path, img_h, img_w):
    x = tf.io.read_file(path)
    x = tf.image.decode_image(x, channels=1, expand_animations=False)
    x = tf.image.resize(x, (img_h, img_w), method='nearest')
    x = tf.cast(x, tf.float32) / 255.0
    x.set_shape([img_h, img_w, 1])
    return x

def tf_parse(x, y, img_h, img_w):
    x = read_image(x, img_h, img_w)
    y = read_mask(y, img_h, img_w)
    return x, y

@tf.function
def augment_data(x, y):
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
    
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
    
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k=k)
    y = tf.image.rot90(y, k=k)
    
    x = tf.image.random_brightness(x, max_delta=0.2)
    x = tf.image.random_contrast(x, lower=0.8, upper=1.2)
    x = tf.clip_by_value(x, 0.0, 1.0)
    
    return x, y

def tf_dataset(X, Y, batch=8, img_h=256, img_w=256, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    
    if augment:
        dataset = dataset.shuffle(buffer_size=min(1000, len(X)))
    
    dataset = dataset.map(
        lambda x, y: tf_parse(x, y, img_h, img_w),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    dataset = dataset.cache()
    
    if augment:
        dataset = dataset.map(
            augment_data,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )
    
    dataset = dataset.batch(batch, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            gc.collect()
            tf.keras.backend.clear_session()

class CosineLRSchedule(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, total_epochs, warmup_epochs=5):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        print(f"\nEpoch {epoch + 1}: Learning rate is {lr:.6f}")

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Argument Parsing """
    parser = argparse.ArgumentParser(description="UNet3+ Training")
    parser.add_argument("--dataset_path", type=str, default="Kvasir-SEG", help="Path to the dataset")
    parser.add_argument("--save_path", type=str, default="files", help="Path to save model and logs")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--img_size", type=int, default=256, help="Image size (H and W)")
    parser.add_argument("--use_cosine_lr", action='store_true', help="Use cosine learning rate schedule")
    parser.add_argument("--mixed_precision", action='store_true', default=True, help="Enable mixed precision training")
    args = parser.parse_args()

    if args.mixed_precision:
        setup_gpu()

    create_dir(args.save_path)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.save_path, f"run_{timestamp}")
    create_dir(run_dir)

    """ Hyperparameters """
    IMG_H = args.img_size
    IMG_W = args.img_size
    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.epochs
    model_path = os.path.join(run_dir, "model.keras")
    csv_path = os.path.join(run_dir, "log.csv")
    tensorboard_dir = os.path.join(run_dir, "tensorboard")

    print("\nTRAINING CONFIGURATION")
    print("-" * 30)
    print(f"Image Size: {IMG_H}x{IMG_W}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    print(f"Epochs: {num_epochs}")
    print("-" * 30 + "\n")

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(args.dataset_path)

    print(f"Train: \t{len(train_x)} images - {len(train_y)} masks")
    print(f"Valid: \t{len(valid_x)} images - {len(valid_y)} masks")
    print(f"Test: \t{len(test_x)} images - {len(test_y)} masks")
    print(f"Steps per epoch: {len(train_x) // batch_size}\n")

    train_dataset = tf_dataset(
        train_x, train_y, 
        batch=batch_size, 
        img_h=IMG_H, 
        img_w=IMG_W, 
        augment=True
    )
    valid_dataset = tf_dataset(
        valid_x, valid_y, 
        batch=batch_size, 
        img_h=IMG_H, 
        img_w=IMG_W, 
        augment=False
    )

    """ Model """
    print("Building UNet3+ model...")
    model = unet3plus((IMG_H, IMG_W, 3))
    
    optimizer = Adam(
        learning_rate=lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
    )
    
    if args.mixed_precision:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        loss=dice_loss,
        optimizer=optimizer,
        metrics=[dice_coef, iou, "Recall", "Precision"],
        jit_compile=True
    )
    
    print(f"Model compiled successfully!")
    print(f"Total parameters: {model.count_params():,}")

    """ Checkpoint & Resume Logic """
    initial_epoch = 0
    if os.path.exists(model_path):
        print(f"\nFound existing model at {model_path}. Loading weights...")
        try:
            model.load_weights(model_path)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Could not load weights: {e}")

    if os.path.exists(csv_path):
        print(f"Found existing logs at {csv_path}. Reading initial epoch...")
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                initial_epoch = df["epoch"].max() + 1
                print(f"Resuming from epoch {initial_epoch + 1}")
        except Exception as e:
            print(f"Could not read logs: {e}")

    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor='val_dice_coef',
            mode='max',
            verbose=1,
            save_best_only=True,
            save_weights_only=False
        ),
        CSVLogger(csv_path, append=True),
        TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch'
        ),
        EarlyStopping(
            monitor='val_dice_coef',
            mode='max',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        MemoryCleanupCallback()
    ]
    
    if args.use_cosine_lr:
        callbacks.append(CosineLRSchedule(lr, num_epochs, warmup_epochs=5))
    else:
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        )

    print("\nSTARTING TRAINING")
    print("-" * 30 + "\n")
    
    try:
        history = model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=valid_dataset,
            callbacks=callbacks,
            initial_epoch=initial_epoch,
            verbose=1
        )
        
        print("\nTRAINING COMPLETED SUCCESSFULLY")
        print("-" * 30)
        
        final_model_path = os.path.join(run_dir, "model_final.keras")
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Model saved at: {model_path}")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        raise
    finally:
        gc.collect()
        tf.keras.backend.clear_session()
