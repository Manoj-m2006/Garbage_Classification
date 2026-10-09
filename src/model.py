"""
model.py
--------
Transfer Learning model architecture module using MobileNetV2 pre-trained on ImageNet.
Provides functions to build initial feature extractor model and unfreeze layers for fine-tuning.
"""

import tensorflow as tf
from typing import Tuple


def build_garbage_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 6,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Builds and compiles MobileNetV2 Transfer Learning model.
    The base MobileNetV2 network is frozen for initial feature extraction.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)

    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="GarbageMobileNetV2")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def unfreeze_for_finetuning(model: tf.keras.Model, fine_tune_at: int = 100, learning_rate: float = 3e-5) -> tf.keras.Model:
    """
    Unfreezes top layers of MobileNetV2 base model for fine-tuning with a low learning rate.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or 'mobilenet' in layer.name.lower():
            base_model = layer
            break

    if base_model is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        print(f"[INFO] MobileNetV2 base model unfrozen from layer {fine_tune_at} onwards for fine-tuning.")
    else:
        print("[WARN] Base model layer not found for unfreezing.")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    model = build_garbage_cnn()
    model.summary()
