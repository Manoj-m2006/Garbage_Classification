"""
evaluate.py
-----------
Module for evaluating trained CNN model on test data split, generating quantitative metrics,
saving classification report JSON, plotting confusion matrix heatmap, and plotting training history curves.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from src.data_preprocessing import CLASSES, prepare_dataset, create_test_dataset


def plot_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix.png"):
    """
    Generates and saves a seaborn confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=True
    )
    plt.title('Garbage Classification CNN - Confusion Matrix', fontsize=14, pad=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, labelpad=8)
    plt.ylabel('True Label', fontsize=12, labelpad=8)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved confusion matrix to '{save_path}'.")


def plot_training_history(history_csv_path="results/training_history.csv", save_path="results/training_history.png"):
    """
    Plots and saves loss and accuracy curves over epochs.
    """
    if not os.path.exists(history_csv_path):
        print(f"[WARN] History CSV file not found at '{history_csv_path}'. Skipping history plots.")
        return

    df = pd.read_csv(history_csv_path)

    plt.figure(figsize=(12, 5), dpi=300)

    # Plot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['accuracy'], label='Train Accuracy', color='#2563eb', linewidth=2)
    plt.plot(df['epoch'], df['val_accuracy'], label='Val Accuracy', color='#059669', linewidth=2, linestyle='--')
    plt.title('Model Accuracy Curves', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Plot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['loss'], label='Train Loss', color='#dc2626', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#d97706', linewidth=2, linestyle='--')
    plt.title('Model Loss Curves', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved training history plots to '{save_path}'.")


def evaluate_model(
    model_path: str = "models/best_garbage_cnn.keras",
    dataset_dir: str = "dataset",
    results_dir: str = "results"
):
    """
    Evaluates saved CNN model on the test dataset split and generates evaluation artifacts.
    """
    os.makedirs(results_dir, exist_ok=True)

    # 1. Prepare test set
    _, _, test_df = prepare_dataset(dataset_dir, random_state=42)
    test_dataset = create_test_dataset(test_df, batch_size=32)

    # 2. Load trained model
    print(f"[INFO] Loading trained model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    # 3. Collect true labels and predictions
    y_true = []
    y_pred_probs = []

    print("[INFO] Running test set evaluation...")
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_pred_probs.append(preds)
        y_true.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(y_true, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Generate Classification Report
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True
    )
    report_str = classification_report(y_true, y_pred, target_names=CLASSES)

    print("\n========================================")
    print("TEST SET EVALUATION RESULTS")
    print("========================================")
    print(report_str)
    print("========================================\n")

    # 5. Save report to JSON
    report_json_path = os.path.join(results_dir, "classification_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"[INFO] Saved classification report JSON to '{report_json_path}'.")

    # 6. Generate plots
    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(results_dir, "confusion_matrix.png"))
    plot_training_history(
        history_csv_path=os.path.join(results_dir, "training_history.csv"),
        save_path=os.path.join(results_dir, "training_history.png")
    )

    return report_dict


if __name__ == "__main__":
    evaluate_model()
