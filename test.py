
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from train import load_dataset, create_dir
import argparse

def read_image(path, img_h, img_w):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (img_w, img_h))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

def read_mask(path, img_h, img_w):
    path = path.decode()
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (img_w, img_h))
    mask = mask / 255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Argument Parsing """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="Kvasir-SEG", help="Path to the dataset")
    parser.add_argument("--model_path", type=str, default="files/model.keras", help="Path to the saved model")
    parser.add_argument("--save_path", type=str, default="results", help="Path to save results")
    parser.add_argument("--img_size", type=int, default=256, help="Image size (H and W)")
    args = parser.parse_args()

    """ Directory for storing files """
    create_dir(args.save_path)

    """ Hyperparameters """
    IMG_H = args.img_size
    IMG_W = args.img_size

    """ Load the model """
    with CustomObjectScope({"dice_loss": dice_loss, "dice_coef": dice_coef, "iou": iou}):
        model = tf.keras.models.load_model(args.model_path)

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(args.dataset_path)

    print(f"Train: \t{len(train_x)} - {len(train_y)}")
    print(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
    print(f"Test: \t{len(test_x)} - {len(test_y)}")

    """ Prediction and Evaluation """
    score = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading the image and mask """
        ori_x = cv2.imread(x, cv2.IMREAD_COLOR)
        ori_x = cv2.resize(ori_x, (IMG_W, IMG_H))
        
        x = ori_x / 255.0
        x = np.expand_dims(x, axis=0)

        ori_y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        ori_y = cv2.resize(ori_y, (IMG_W, IMG_H))
        ori_y = ori_y / 255.0
        ori_y = np.expand_dims(ori_y, axis=-1)

        """ Prediction """
        pred = model.predict(x, verbose=0)[0]
        pred_mask = (pred > 0.5).astype(np.int32)

        """ Save final mask """
        save_image_path = os.path.join(args.save_path, f"{name}.jpg")
        
        # Concatenate: Original Image | Ground Truth | Predicted Mask
        sep_line = np.ones((IMG_H, 10, 3)) * 255
        
        # Ensure strict shapes for concatenation
        cat_images = np.concatenate([
            ori_x, 
            sep_line, 
            np.concatenate([ori_y, ori_y, ori_y], axis=-1) * 255, 
            sep_line, 
            np.concatenate([pred_mask, pred_mask, pred_mask], axis=-1) * 255
        ], axis=1)
        
        cv2.imwrite(save_image_path, cat_images)

    """ Evaluation on Test Set """
    # Create TF Dataset for batched evaluation
    def tf_parse_test(x, y):
        def _parse(x, y):
            x = read_image(x, IMG_H, IMG_W)
            y = read_mask(y, IMG_H, IMG_W)
            return x, y
        x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
        x.set_shape([IMG_H, IMG_W, 3])
        y.set_shape([IMG_H, IMG_W, 1])
        return x, y

    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_dataset = test_dataset.map(tf_parse_test).batch(2) # Batch size 2 like training

    print("\nEvaluating Model...")
    results = model.evaluate(test_dataset)
    
    print("\nModel Metrics:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")
