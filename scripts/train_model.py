import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ASSETS_DIR,
    CLASS_NAMES,
    CLASS_NAMES_PATH,
    CONFUSION_MATRIX_PATH,
    MODEL_DIR,
    MODEL_INPUT_SIZE,
    MODEL_METRICS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TEST_SPLIT,
    TRAINING_EPOCHS,
    TRAINING_HISTORY_PATH,
    TRAINING_SAMPLES_PER_CLASS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SketchGuess AI model.")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=TRAINING_SAMPLES_PER_CLASS,
        help="Number of drawings to load from each class.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_EPOCHS,
        help="Number of training epochs.",
    )
    return parser.parse_args()


def check_data_files():
    missing_files = [
        RAW_DATA_DIR / f"{class_name}.npy"
        for class_name in CLASS_NAMES
        if not (RAW_DATA_DIR / f"{class_name}.npy").exists()
    ]

    if missing_files:
        print("Missing Quick, Draw! files:")
        for file_path in missing_files:
            print(f"- {file_path}")
        print("\nRun this first:")
        print("python scripts/download_quickdraw_data.py")
        raise SystemExit(1)


def load_dataset(samples_per_class):
    images = []
    labels = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        file_path = RAW_DATA_DIR / f"{class_name}.npy"
        class_data = np.load(file_path, mmap_mode="r")
        sample_count = min(samples_per_class, len(class_data))

        print(f"Loading {sample_count} samples for '{class_name}'")

        class_images = np.array(class_data[:sample_count], dtype="float32")
        class_images = class_images.reshape(
            sample_count,
            MODEL_INPUT_SIZE[0],
            MODEL_INPUT_SIZE[1],
            1,
        )
        class_images = class_images / 255.0

        class_labels = np.full(sample_count, class_index, dtype="int64")

        images.append(class_images)
        labels.append(class_labels)

    return np.concatenate(images), np.concatenate(labels)


def split_dataset(images, labels):
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_indices = rng.permutation(len(images))
    test_count = int(len(images) * TEST_SPLIT)

    test_indices = shuffled_indices[:test_count]
    train_indices = shuffled_indices[test_count:]

    return (
        images[train_indices],
        images[test_indices],
        labels[train_indices],
        labels[test_indices],
    )


def build_model(num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 1)),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_class_names():
    CLASS_NAMES_PATH.write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    print(f"Saved class names to {CLASS_NAMES_PATH}")


def save_training_history_chart(history):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.history["accuracy"], label="Training accuracy")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history.history["loss"], label="Training loss")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(TRAINING_HISTORY_PATH, dpi=150)
    plt.close(fig)

    print(f"Saved training history chart to {TRAINING_HISTORY_PATH}")


def save_confusion_matrix_image(y_true, y_pred):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype="int32")

    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[true_label, predicted_label] += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    for row in range(len(CLASS_NAMES)):
        for col in range(len(CLASS_NAMES)):
            value = matrix[row, col]
            if value > 0:
                ax.text(col, row, str(value), ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)

    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")


def save_metrics(samples_per_class, epochs, test_loss, test_accuracy):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "number_of_classes": len(CLASS_NAMES),
        "samples_per_class": samples_per_class,
        "epochs": epochs,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
    }

    MODEL_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to {MODEL_METRICS_PATH}")


def main():
    args = parse_args()

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    check_data_files()

    images, labels = load_dataset(args.samples_per_class)
    x_train, x_test, y_train, y_test = split_dataset(images, labels)

    print(f"\nTraining images: {len(x_train)}")
    print(f"Test images: {len(x_test)}")

    model = build_model(num_classes=len(CLASS_NAMES))
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=128,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("\nEvaluation complete")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Test loss: {test_loss:.4f}")

    test_probabilities = model.predict(x_test, batch_size=128, verbose=0)
    predicted_labels = np.argmax(test_probabilities, axis=1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    save_class_names()
    save_metrics(args.samples_per_class, args.epochs, test_loss, test_accuracy)
    save_training_history_chart(history)
    save_confusion_matrix_image(y_test, predicted_labels)

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
