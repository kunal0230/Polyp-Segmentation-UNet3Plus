
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision # Added import
from model import unet3plus
from metrics import dice_loss, dice_coef, iou
import argparse

# Enable Mixed Precision for T4 GPU (speeds up training)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.1):
    """ Loading the images and masks """
    X = sorted(glob(os.path.join(path, "images", "*")))
    Y = sorted(glob(os.path.join(path, "masks", "*")))

    """ Spliting the data into training and testing """
    split_size = max(1, int(len(X) * split))  # Ensure split_size is at least 1

    train_x, valid_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(Y, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path, img_h, img_w):
    x = tf.io.read_file(path)
    x = tf.image.decode_image(x, channels=3, expand_animations=False)
    x = tf.image.resize(x, (img_h, img_w))
    x = x / 255.0
    x.set_shape([img_h, img_w, 3])
    return x

def read_mask(path, img_h, img_w):
    x = tf.io.read_file(path)
    x = tf.image.decode_image(x, channels=1, expand_animations=False)
    x = tf.image.resize(x, (img_h, img_w))
    x = x / 255.0
    x.set_shape([img_h, img_w, 1])
    return x

def tf_parse(x, y, img_h, img_w):
    x = read_image(x, img_h, img_w)
    y = read_mask(y, img_h, img_w)
    return x, y

def tf_dataset(X, Y, batch=8, img_h=256, img_w=256, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    
    def _augment_fn(x, y):
        # Random horizontal flip
        if tf.random.uniform(()) > 0.5:
            x = tf.image.flip_left_right(x)
            y = tf.image.flip_left_right(y)
        # Random vertical flip
        if tf.random.uniform(()) > 0.5:
            x = tf.image.flip_up_down(x)
            y = tf.image.flip_up_down(y)
        return x, y

    ds = ds.map(lambda x, y: tf_parse(x, y, img_h, img_w), num_parallel_calls=tf.data.AUTOTUNE)

    # Use cache to load data into memory for faster training
    ds = ds.cache()

    if augment:
        ds = ds.map(_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Argument Parsing """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="Kvasir-SEG", help="Path to the dataset")
    parser.add_argument("--save_path", type=str, default="files", help="Path to save model and logs")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=256, help="Image size (H and W)")
    args = parser.parse_args()

    """ Directory for storing files """
    create_dir(args.save_path)

    """ Hyperparameters """
    IMG_H = args.img_size
    IMG_W = args.img_size
    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.epochs
    model_path = os.path.join(args.save_path, "model.keras")
    csv_path = os.path.join(args.save_path, "log.csv")

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(args.dataset_path)

    print(f"Train: \t{len(train_x)} - {len(train_y)}")
    print(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
    print(f"Test: \t{len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size, img_h=IMG_H, img_w=IMG_W, augment=True)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size, img_h=IMG_H, img_w=IMG_W, augment=False)

    """ Model """
    model = unet3plus((IMG_H, IMG_W, 3))
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, "Recall", "Precision"])
    # model.summary()

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-10, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
