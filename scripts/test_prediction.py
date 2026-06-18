import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ASSETS_DIR,
    CLASS_NAMES_PATH,
    MODEL_INPUT_SIZE,
    MODEL_PATH,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SAMPLE_PREDICTIONS_PATH,
    TOP_K_PREDICTIONS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test saved SketchGuess AI predictions.")
    parser.add_argument(
        "--samples",
        type=int,
        default=6,
        help="Number of random samples to test.",
    )
    return parser.parse_args()


def load_artifacts():
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        print("Saved model files were not found.")
        print("Run this first:")
        print("python scripts/train_model.py")
        raise SystemExit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return model, class_names


def load_random_samples(class_names, sample_count):
    rng = np.random.default_rng(RANDOM_SEED)
    samples = []

    for _ in range(sample_count):
        class_index = int(rng.integers(0, len(class_names)))
        class_name = class_names[class_index]
        file_path = RAW_DATA_DIR / f"{class_name}.npy"

        if not file_path.exists():
            print(f"Missing raw data file: {file_path}")
            print("Run this first:")
            print("python scripts/download_quickdraw_data.py")
            raise SystemExit(1)

        class_data = np.load(file_path, mmap_mode="r")
        sample_index = int(rng.integers(0, len(class_data)))
        image_pixels = np.array(class_data[sample_index], dtype="float32")
        image_pixels = image_pixels.reshape(MODEL_INPUT_SIZE)

        samples.append(
            {
                "true_label": class_name,
                "image_pixels": image_pixels,
            }
        )

    return samples


def predict_sample(model, class_names, image_pixels):
    image_array = image_pixels.reshape(1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 1)
    image_array = image_array / 255.0

    probabilities = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(probabilities)[-TOP_K_PREDICTIONS:][::-1]

    return [
        (class_names[index], float(probabilities[index]))
        for index in top_indices
    ]


def print_predictions(samples):
    for index, sample in enumerate(samples, start=1):
        print(f"\nSample {index}: true label = {sample['true_label']}")

        for rank, (label, confidence) in enumerate(sample["predictions"], start=1):
            print(f"  {rank}. {label}: {confidence:.2%}")


def save_prediction_preview(samples):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    cell_width = 210
    cell_height = 145
    image_size = 84
    columns = 3
    rows = int(np.ceil(len(samples) / columns))

    preview = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()

    for index, sample in enumerate(samples):
        col = index % columns
        row = index // columns
        x = col * cell_width
        y = row * cell_height

        display_pixels = 255 - sample["image_pixels"].astype("uint8")
        thumbnail = Image.fromarray(display_pixels, mode="L")
        thumbnail = thumbnail.resize((image_size, image_size), Image.Resampling.NEAREST)
        thumbnail = thumbnail.convert("RGB")

        preview.paste(thumbnail, (x + 8, y + 8))

        top_label, top_confidence = sample["predictions"][0]
        draw.text((x + 100, y + 12), f"True: {sample['true_label']}", fill="black", font=font)
        draw.text((x + 100, y + 32), f"Top: {top_label}", fill="black", font=font)
        draw.text((x + 100, y + 52), f"{top_confidence:.1%}", fill="black", font=font)

        y_offset = y + 78
        for rank, (label, confidence) in enumerate(sample["predictions"], start=1):
            draw.text(
                (x + 8, y_offset),
                f"{rank}. {label} ({confidence:.1%})",
                fill="black",
                font=font,
            )
            y_offset += 16

    preview.save(SAMPLE_PREDICTIONS_PATH)
    print(f"\nSaved sample prediction preview to {SAMPLE_PREDICTIONS_PATH}")


def main():
    args = parse_args()
    model, class_names = load_artifacts()

    samples = load_random_samples(class_names, args.samples)

    for sample in samples:
        sample["predictions"] = predict_sample(
            model,
            class_names,
            sample["image_pixels"],
        )

    print_predictions(samples)
    save_prediction_preview(samples)


if __name__ == "__main__":
    main()
