"""
data_preprocessing.py
---------------------
Module responsible for dataset discovery, stratified splitting (80/10/10),
image preprocessing with MobileNetV2 input scaling, data augmentation,
tf.data.Dataset creation with in-memory caching, and balanced class weight calculation.
"""

import os
from pathlib import Path
from typing import List, Tuple, Union, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASS_TO_INDEX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def discover_images(dataset_dir: Union[str, Path] = "dataset") -> List[Tuple[str, str, int]]:
    """Recursively discovers image files from dataset_dir filtering out OS metadata."""
    root_path = Path(dataset_dir).resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Dataset directory '{root_path}' does not exist.")

    images = []
    
    for cls_name in CLASSES:
        cls_dirs = []
        direct_dir = root_path / cls_name
        nested_dir = root_path / "dataset-resized" / cls_name
        if direct_dir.is_dir():
            cls_dirs.append(direct_dir)
        elif nested_dir.is_dir():
            cls_dirs.append(nested_dir)
        else:
            found = [
                d for d in root_path.rglob("*")
                if d.is_dir() and d.name.lower() == cls_name.lower() and "__MACOSX" not in d.parts
            ]
            cls_dirs.extend(found)
            
        if not cls_dirs:
            raise ValueError(f"Validation Error: Missing directory for expected class '{cls_name}'.")

        cls_image_count = 0
        for d in cls_dirs:
            for filepath in d.rglob("*"):
                if "__MACOSX" in filepath.parts or filepath.name.startswith("._") or filepath.name.startswith("."):
                    continue
                if filepath.is_file() and filepath.suffix.lower() in VALID_EXTENSIONS:
                    images.append((str(filepath), cls_name, CLASS_TO_INDEX[cls_name]))
                    cls_image_count += 1
                    
        if cls_image_count == 0:
            raise ValueError(f"Validation Error: No valid images found for class '{cls_name}'.")

    return images


def create_dataframe(images: List[Tuple[str, str, int]]) -> pd.DataFrame:
    """Creates a pandas DataFrame with columns ['filepath', 'label', 'class_index']."""
    df = pd.DataFrame(images, columns=['filepath', 'label', 'class_index'])

    if df['filepath'].isnull().any() or df['label'].isnull().any() or df['class_index'].isnull().any():
        raise ValueError("Validation Error: Found images with null or unassigned labels.")

    present_classes = set(df['label'].unique())
    missing = set(CLASSES) - present_classes
    if missing:
        raise ValueError(f"Validation Error: The following class(es) are missing: {missing}")

    return df


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Performs stratified split into train (80%), val (10%), test (10%)."""
    val_test_ratio = val_ratio + test_ratio  # 0.20
    
    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=df['class_index']
    )

    test_relative_ratio = test_ratio / val_test_ratio  # 0.50
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_relative_ratio,
        random_state=random_state,
        stratify=temp_df['class_index']
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_dataset(
    dataset_dir: Union[str, Path] = "dataset",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pipeline function to discover images, create DataFrame, and return stratified splits."""
    images = discover_images(dataset_dir)
    df = create_dataframe(images)
    return stratified_split(df, random_state=random_state)


def _parse_image(filepath: str, label: int, image_size: Tuple[int, int] = IMAGE_SIZE):
    """
    Reads image file, decodes RGB, resizes to (224, 224), applies MobileNetV2 [-1, 1] scaling.
    """
    img_bytes = tf.io.read_file(filepath)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    # Apply MobileNetV2 preprocessing [-1.0, 1.0]
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    label_one_hot = tf.one_hot(label, depth=len(CLASSES))
    return img, label_one_hot


def get_augmentation_model() -> tf.keras.Sequential:
    """Data augmentation pipeline for training."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(factor=0.10),
        tf.keras.layers.RandomZoom(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)),
        tf.keras.layers.RandomTranslation(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05)),
        tf.keras.layers.RandomContrast(factor=0.1)
    ], name="training_augmentation")


def create_train_dataset(
    train_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE
) -> tf.data.Dataset:
    """Creates tf.data.Dataset for training set with caching and augmentation."""
    filepaths = train_df['filepath'].values
    labels = train_df['class_index'].values

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(
        lambda fp, lbl: _parse_image(fp, lbl, image_size=image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=len(train_df), seed=42)
    dataset = dataset.batch(batch_size)

    augmentation_layer = get_augmentation_model()
    dataset = dataset.map(
        lambda x, y: (augmentation_layer(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def create_validation_dataset(
    val_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE
) -> tf.data.Dataset:
    """Creates tf.data.Dataset for validation set."""
    filepaths = val_df['filepath'].values
    labels = val_df['class_index'].values

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(
        lambda fp, lbl: _parse_image(fp, lbl, image_size=image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def create_test_dataset(
    test_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE
) -> tf.data.Dataset:
    """Creates tf.data.Dataset for test set."""
    filepaths = test_df['filepath'].values
    labels = test_df['class_index'].values

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(
        lambda fp, lbl: _parse_image(fp, lbl, image_size=image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def calculate_class_weights(train_df: pd.DataFrame) -> Dict[int, float]:
    """Calculates balanced class weights from training set labels."""
    train_labels = train_df['class_index'].values
    unique_classes = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_labels
    )
    return {int(cls_idx): float(w) for cls_idx, w in zip(unique_classes, weights)}
