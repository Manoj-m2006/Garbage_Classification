"""
train.py
--------
Training pipeline module for Garbage Classification Transfer Learning Model (MobileNetV2).
Implements a 2-phase training strategy:
1. Feature Extraction: Train top dense head with frozen MobileNetV2 base.
2. Fine-Tuning: Unfreeze top layers of MobileNetV2 with low learning rate.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import tensorflow as tf

from src.data_preprocessing import (
    CLASSES,
    prepare_dataset,
    create_train_dataset,
    create_validation_dataset,
    calculate_class_weights
)
from src.model import build_garbage_cnn, unfreeze_for_finetuning


class LearningRateLogger(tf.keras.callbacks.Callback):
    """Custom Keras callback to record learning rate at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            optimizer = self.model.optimizer
            lr = optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                lr_val = float(lr.numpy())
            elif callable(lr):
                lr_val = float(lr(optimizer.iterations))
            else:
                lr_val = float(lr)
            logs['learning_rate'] = lr_val


def train_model(
    dataset_dir: str = "dataset",
    batch_size: int = 32,
    phase1_epochs: int = 10,
    phase2_epochs: int = 10,
    model_save_path: str = "models/best_garbage_cnn.keras",
    history_save_path: str = "results/training_history.csv"
):
    """
    Executes 2-phase Transfer Learning training loop: Feature Extraction followed by Fine-Tuning.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_save_path), exist_ok=True)

    if os.path.exists(model_save_path):
        try:
            os.remove(model_save_path)
        except Exception as e:
            print(f"[WARN] Could not remove existing model file: {e}")

    # 1. Load dataset splits
    train_df, val_df, test_df = prepare_dataset(dataset_dir, random_state=42)

    # 2. Build tf.data datasets
    train_dataset = create_train_dataset(train_df, batch_size=batch_size)
    val_dataset = create_validation_dataset(val_df, batch_size=batch_size)

    # 3. Calculate class weights
    class_weights = calculate_class_weights(train_df)

    print("========================================")
    print("GARBAGE CLASSIFICATION MOBILENETV2 TRAINING")
    print("========================================")
    print(f"Training samples   : {len(train_df)}")
    print(f"Validation samples : {len(val_df)}")
    print(f"Test samples       : {len(test_df)}")
    print(f"Batch size         : {batch_size}")
    print(f"Phase 1 Epochs     : {phase1_epochs}")
    print(f"Phase 2 Epochs     : {phase2_epochs}")
    print("\nClass weights:")
    for cls_idx, cls_name in enumerate(CLASSES):
        print(f"  {cls_name:<10} : {class_weights[cls_idx]:.4f}")
    print("========================================\n")

    # 4. Build MobileNetV2 Feature Extraction Model
    model = build_garbage_cnn(input_shape=(224, 224, 3), num_classes=6, learning_rate=0.001)

    callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=2, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_loss', mode='min', save_best_only=True),
        LearningRateLogger()
    ]

    print("--- PHASE 1: FEATURE EXTRACTION (FROZEN BACKBONE) ---")
    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=phase1_epochs,
        class_weight=class_weights,
        callbacks=callbacks_phase1
    )

    print("\n--- PHASE 2: FINE-TUNING (UNFROZEN TOP LAYERS) ---")
    model = unfreeze_for_finetuning(model, fine_tune_at=100, learning_rate=3e-5)

    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_loss', mode='min', save_best_only=True),
        LearningRateLogger()
    ]

    history2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=phase1_epochs + phase2_epochs,
        initial_epoch=history1.epoch[-1] + 1 if history1.epoch else phase1_epochs,
        class_weight=class_weights,
        callbacks=callbacks_phase2
    )

    # 5. Merge and Save History
    h1 = history1.history
    h2 = history2.history

    combined_loss = h1['loss'] + h2['loss']
    combined_acc = h1['accuracy'] + h2['accuracy']
    combined_val_loss = h1['val_loss'] + h2['val_loss']
    combined_val_acc = h1['val_accuracy'] + h2['val_accuracy']
    
    lr1 = h1.get('learning_rate', h1.get('lr', [0.001] * len(h1['loss'])))
    lr2 = h2.get('learning_rate', h2.get('lr', [3e-5] * len(h2['loss'])))
    combined_lr = list(lr1) + list(lr2)

    total_epochs = len(combined_loss)
    history_df = pd.DataFrame({
        'epoch': list(range(1, total_epochs + 1)),
        'loss': combined_loss,
        'accuracy': combined_acc,
        'val_loss': combined_val_loss,
        'val_accuracy': combined_val_acc,
        'learning_rate': combined_lr
    })

    history_df.to_csv(history_save_path, index=False)

    best_val_loss = min(combined_val_loss)
    best_val_loss_idx = combined_val_loss.index(best_val_loss)
    best_val_acc = combined_val_acc[best_val_loss_idx]

    print("\n========================================")
    print("TRANSFER LEARNING TRAINING COMPLETE")
    print("========================================")
    print(f"Total epochs trained     : {total_epochs}")
    print(f"Best Validation Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Best Validation Loss     : {best_val_loss:.4f}")
    print(f"Saved model path         : {model_save_path}")
    print(f"Saved history CSV path   : {history_save_path}")
    print("========================================\n")
    return model, history_df


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    train_model()
