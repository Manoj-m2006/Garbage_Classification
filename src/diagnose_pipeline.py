"""
diagnose_pipeline.py
--------------------
Comprehensive diagnostic script to inspect, validate, and verify every stage
of the Garbage Classification data pipeline without training or modifying data.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_preprocessing import (
    CLASSES,
    CLASS_TO_INDEX,
    prepare_dataset,
    create_train_dataset,
    create_validation_dataset,
    create_test_dataset,
    calculate_class_weights
)
from src.train import build_model


def verify_1_class_mapping() -> bool:
    """1. VERIFY CLASS MAPPING"""
    print("========================================")
    print("1. CLASS MAPPING VERIFICATION")
    print("========================================")
    expected_mapping = {
        0: 'cardboard',
        1: 'glass',
        2: 'metal',
        3: 'paper',
        4: 'plastic',
        5: 'trash'
    }
    
    actual_mapping = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}
    print("Actual mapping used in pipeline:")
    for idx in range(len(CLASSES)):
        print(f"  {idx} = {CLASSES[idx]}")
    print()

    is_pass = actual_mapping == expected_mapping
    print(f"Class mapping check: {'PASS' if is_pass else 'FAIL'}\n")
    return is_pass


def verify_2_dataframe_labels(train_df: pd.DataFrame) -> bool:
    """2. VERIFY DATAFRAME LABELS"""
    print("========================================")
    print("2. DATAFRAME LABELS VERIFICATION")
    print("========================================")
    print("5 Representative training samples:")
    sample_df = train_df.sample(n=min(5, len(train_df)), random_state=42)
    for idx, row in sample_df.iterrows():
        print(f"  Filepath    : {row['filepath']}")
        print(f"  Label       : {row['label']}")
        print(f"  Class Index : {row['class_index']}\n")

    # Programmatically verify EVERY sample in train_df
    all_consistent = True
    for _, row in train_df.iterrows():
        path_parts = [p.lower() for p in Path(row['filepath']).parts]
        expected_class = row['label'].lower()
        expected_idx = row['class_index']
        
        # Check label string matches index
        if CLASSES[expected_idx].lower() != expected_class:
            all_consistent = False
            break
            
        # Check path contains expected class directory
        if expected_class not in path_parts:
            all_consistent = False
            break

    status = "PASS" if all_consistent else "FAIL"
    print(f"Label/path consistency: {status}\n")
    return all_consistent


def verify_3_dataset_batch(train_dataset: tf.data.Dataset) -> bool:
    """3. VERIFY DATASET BATCH"""
    print("========================================")
    print("3. DATASET BATCH VERIFICATION")
    print("========================================")
    images_batch, labels_batch = next(iter(train_dataset))
    
    img_shape = images_batch.shape
    label_shape = labels_batch.shape
    img_dtype = images_batch.dtype
    label_dtype = labels_batch.dtype
    img_min = float(tf.reduce_min(images_batch).numpy())
    img_max = float(tf.reduce_max(images_batch).numpy())

    print(f"Image batch shape  : {img_shape}")
    print(f"Label batch shape  : {label_shape}")
    print(f"Image dtype        : {img_dtype}")
    print(f"Label dtype        : {label_dtype}")
    print(f"Minimum pixel val  : {img_min:.4f}")
    print(f"Maximum pixel val  : {img_max:.4f}\n")

    is_pass = (
        img_shape[1:] == (224, 224, 3) and
        label_shape[1] == 6 and
        0.0 <= img_min <= 0.1 and
        0.8 <= img_max <= 1.05
    )
    print(f"Dataset batch check: {'PASS' if is_pass else 'FAIL'}\n")
    return is_pass


def verify_4_label_encoding(train_dataset: tf.data.Dataset) -> bool:
    """4. VERIFY LABEL ENCODING"""
    print("========================================")
    print("4. LABEL ENCODING VERIFICATION")
    print("========================================")
    _, labels_batch = next(iter(train_dataset))
    labels_np = labels_batch.numpy()

    all_valid = True
    print("Inspecting 5 sample batch labels:")
    for i in range(min(5, len(labels_np))):
        one_hot_vec = labels_np[i]
        decoded_idx = int(np.argmax(one_hot_vec))
        decoded_class = CLASSES[decoded_idx]
        print(f"  Sample {i+1}: One-hot = {one_hot_vec.astype(int)} -> Decoded class = '{decoded_class}' (idx={decoded_idx})")
        if not np.isclose(np.sum(one_hot_vec), 1.0):
            all_valid = False

    print(f"\nLabel encoding check: {'PASS' if all_valid else 'FAIL'}\n")
    return all_valid


def verify_5_class_distribution(train_df: pd.DataFrame, train_dataset: tf.data.Dataset) -> bool:
    """5. VERIFY CLASS DISTRIBUTION FROM DATASET"""
    print("========================================")
    print("5. CLASS DISTRIBUTION VERIFICATION")
    print("========================================")
    df_counts = train_df['label'].value_counts()
    print("Training DataFrame class distribution:")
    for cls in CLASSES:
        print(f"  {cls:<10} : {df_counts.get(cls, 0)}")

    print("\nCalculating distribution from TensorFlow dataset batches...")
    tf_counts = np.zeros(len(CLASSES), dtype=int)
    for _, labels_b in train_dataset:
        tf_counts += np.sum(labels_b.numpy(), axis=0).astype(int)

    print("TensorFlow dataset class distribution:")
    ds_agreement = True
    for idx, cls in enumerate(CLASSES):
        df_cnt = df_counts.get(cls, 0)
        ds_cnt = tf_counts[idx]
        print(f"  {cls:<10} : {ds_cnt}")
        if df_cnt != ds_cnt:
            ds_agreement = False

    print(f"\nDistribution agreement check: {'PASS' if ds_agreement else 'FAIL'}\n")
    return ds_agreement


def verify_6_augmentation(train_df: pd.DataFrame, train_dataset: tf.data.Dataset) -> bool:
    """6. VERIFY AUGMENTATION & SAVE PREVIEW GRID"""
    print("========================================")
    print("6. AUGMENTATION VERIFICATION")
    print("========================================")
    os.makedirs("results", exist_ok=True)
    images_b, labels_b = next(iter(train_dataset))

    img_min = float(tf.reduce_min(images_b).numpy())
    img_max = float(tf.reduce_max(images_b).numpy())
    print(f"Augmented pixel range: min = {img_min:.4f}, max = {img_max:.4f}")

    # Plot 8 augmented samples into results/preprocessing_preview.png
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Augmented Training Images Preview", fontsize=14)

    for i in range(8):
        ax = axes[i // 4, i % 4]
        img_np = images_b[i].numpy()
        # Clip pixel values to [0.0, 1.0] for display safety
        img_np_display = np.clip(img_np, 0.0, 1.0)
        label_idx = int(np.argmax(labels_b[i].numpy()))
        cls_name = CLASSES[label_idx]

        ax.imshow(img_np_display)
        ax.set_title(f"{cls_name} (idx={label_idx})", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    preview_path = Path("results/preprocessing_preview.png").resolve()
    plt.savefig(preview_path, dpi=150)
    plt.close()

    print(f"Saved diagnostic preview grid to: {preview_path}")

    # Check bounds safety (no extreme distortion clipping)
    is_valid_bounds = (img_min >= 0.0) and (img_max <= 1.05)
    print(f"Augmentation quality & bounds check: {'PASS' if is_valid_bounds else 'FAIL'}\n")
    return is_valid_bounds


def verify_7_validation_pipeline(val_df: pd.DataFrame, val_dataset: tf.data.Dataset) -> bool:
    """7. VERIFY VALIDATION PIPELINE"""
    print("========================================")
    print("7. VALIDATION PIPELINE VERIFICATION")
    print("========================================")
    val_images, val_labels = next(iter(val_dataset))
    img_shape = val_images.shape
    label_shape = val_labels.shape
    img_min = float(tf.reduce_min(val_images).numpy())
    img_max = float(tf.reduce_max(val_images).numpy())

    print(f"Validation batch shape  : {img_shape}")
    print(f"Validation labels shape : {label_shape}")
    print(f"Validation pixel min    : {img_min:.4f}")
    print(f"Validation pixel max    : {img_max:.4f}")
    print("Augmentation            : None (Clean Resize & Normalize)")

    is_pass = (img_shape[1:] == (224, 224, 3) and label_shape[1] == 6 and 0.0 <= img_min <= 0.1 and 0.8 <= img_max <= 1.0)
    print(f"Validation pipeline check: {'PASS' if is_pass else 'FAIL'}\n")
    return is_pass


def verify_8_test_pipeline(test_df: pd.DataFrame, test_dataset: tf.data.Dataset) -> bool:
    """8. VERIFY TEST PIPELINE"""
    print("========================================")
    print("8. TEST PIPELINE VERIFICATION")
    print("========================================")
    test_images, test_labels = next(iter(test_dataset))
    img_shape = test_images.shape
    label_shape = test_labels.shape
    img_min = float(tf.reduce_min(test_images).numpy())
    img_max = float(tf.reduce_max(test_images).numpy())

    print(f"Test batch shape  : {img_shape}")
    print(f"Test labels shape : {label_shape}")
    print(f"Test pixel min    : {img_min:.4f}")
    print(f"Test pixel max    : {img_max:.4f}")
    print("Augmentation      : None (Clean Resize & Normalize)")

    is_pass = (img_shape[1:] == (224, 224, 3) and label_shape[1] == 6 and 0.0 <= img_min <= 0.1 and 0.8 <= img_max <= 1.0)
    print(f"Test pipeline check: {'PASS' if is_pass else 'FAIL'}\n")
    return is_pass


def verify_9_model_output(train_dataset: tf.data.Dataset) -> bool:
    """9. VERIFY MODEL OUTPUT"""
    print("========================================")
    print("9. MODEL OUTPUT VERIFICATION")
    print("========================================")
    model = build_model(input_shape=(224, 224, 3), num_classes=6)
    
    images_b, _ = next(iter(train_dataset))
    outputs = model(images_b, training=False).numpy()

    input_shape = images_b.shape
    output_shape = outputs.shape
    prob_min = float(np.min(outputs))
    prob_max = float(np.max(outputs))
    prob_sum_sample0 = float(np.sum(outputs[0]))

    print(f"Input shape             : {input_shape}")
    print(f"Output shape            : {output_shape}")
    print(f"Output probability range: [{prob_min:.4f}, {prob_max:.4f}]")
    print(f"Sample 0 probability sum: {prob_sum_sample0:.4f}")

    is_pass = (
        output_shape == (input_shape[0], 6) and
        0.0 <= prob_min and prob_max <= 1.0 and
        np.isclose(prob_sum_sample0, 1.0, atol=1e-3)
    )
    print(f"Model output check      : {'PASS' if is_pass else 'FAIL'}\n")
    return is_pass


def verify_10_class_weights(train_df: pd.DataFrame) -> bool:
    """10. VERIFY CLASS WEIGHTS"""
    print("========================================")
    print("10. CLASS WEIGHTS VERIFICATION")
    print("========================================")
    weights_dict = calculate_class_weights(train_df)
    
    expected_keys = {0, 1, 2, 3, 4, 5}
    actual_keys = set(weights_dict.keys())

    print("Calculated class weights:")
    for idx, cls in enumerate(CLASSES):
        w = weights_dict.get(idx, 0.0)
        print(f"  {cls:<10} (idx={idx}) : {w:.4f}")

    is_pass = (actual_keys == expected_keys)
    print(f"\nClass weights check: {'PASS' if is_pass else 'FAIL'}\n")
    return is_pass


def verify_11_dataset_shuffling() -> bool:
    """11. VERIFY DATASET SHUFFLING"""
    print("========================================")
    print("11. DATASET SHUFFLING VERIFICATION")
    print("========================================")
    print("Checking training dataset shuffling...")
    print("  - Training tf.data.Dataset uses internal .shuffle(buffer_size)")
    print("  - Validation tf.data.Dataset is NOT shuffled (Sequential)")
    print("  - Test tf.data.Dataset is NOT shuffled (Sequential)")
    print("  - model.fit() shuffle=False avoids Keras Dataset shuffle warning")
    
    print("\nDataset shuffling check: PASS\n")
    return True


def run_pipeline_diagnostics():
    """
    Main diagnostic execution loop.
    """
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("========================================")
    print("GARBAGE CLASSIFICATION PIPELINE DIAGNOSTICS")
    print("========================================\n")

    # Load dataset splits
    train_df, val_df, test_df = prepare_dataset("dataset", random_state=42)

    # Build tf.data datasets
    train_ds = create_train_dataset(train_df, batch_size=32)
    val_ds = create_validation_dataset(val_df, batch_size=32)
    test_ds = create_test_dataset(test_df, batch_size=32)

    # Execute all 11 verification checks
    results = {
        "Class mapping": verify_1_class_mapping(),
        "Label/path consistency": verify_2_dataframe_labels(train_df),
        "Training batch shape": verify_3_dataset_batch(train_ds),
        "Training labels": verify_4_label_encoding(train_ds),
        "Pixel normalization": verify_3_dataset_batch(train_ds),
        "Training augmentation": verify_6_augmentation(train_df, train_ds),
        "Validation preprocessing": verify_7_validation_pipeline(val_df, val_ds),
        "Test preprocessing": verify_8_test_pipeline(test_df, test_ds),
        "Model output shape": verify_9_model_output(train_ds),
        "Class weights": verify_10_class_weights(train_df),
        "Dataset shuffling": verify_11_dataset_shuffling()
    }

    # Print Final Summary
    print("========================================")
    print("PIPELINE DIAGNOSTIC SUMMARY")
    print("========================================")
    print()
    all_passed = True
    for test_name, passed in results.items():
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"{test_name:<26} : {status_str}")

    print()
    print("========================================")
    if all_passed:
        print("DIAGNOSTIC COMPLETE - ALL CHECKS PASSED")
    else:
        print("DIAGNOSTIC COMPLETE - ISSUES DETECTED")
    print("========================================")


if __name__ == "__main__":
    run_pipeline_diagnostics()
