import numpy as np
from PIL import Image

from src.config import CROP_PADDING, EMPTY_CANVAS_THRESHOLD, MODEL_INPUT_SIZE


def canvas_to_grayscale(image_data):
    if image_data is None:
        return None

    image = Image.fromarray(image_data.astype("uint8")).convert("L")
    return np.array(image)


def is_canvas_empty(image_data):
    grayscale_pixels = canvas_to_grayscale(image_data)

    if grayscale_pixels is None:
        return True

    return not np.any(grayscale_pixels < EMPTY_CANVAS_THRESHOLD)


def crop_drawing(grayscale_pixels):
    drawing_pixels = np.where(grayscale_pixels < EMPTY_CANVAS_THRESHOLD)

    if len(drawing_pixels[0]) == 0:
        return grayscale_pixels

    y_min = max(drawing_pixels[0].min() - CROP_PADDING, 0)
    y_max = min(drawing_pixels[0].max() + CROP_PADDING, grayscale_pixels.shape[0] - 1)
    x_min = max(drawing_pixels[1].min() - CROP_PADDING, 0)
    x_max = min(drawing_pixels[1].max() + CROP_PADDING, grayscale_pixels.shape[1] - 1)

    return grayscale_pixels[y_min : y_max + 1, x_min : x_max + 1]


def preprocess_canvas_image(image_data):
    if is_canvas_empty(image_data):
        return None

    grayscale_pixels = canvas_to_grayscale(image_data)
    cropped_pixels = crop_drawing(grayscale_pixels)

    image = Image.fromarray(cropped_pixels)
    image = image.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)

    pixels = np.array(image).astype("float32")
    pixels = 1.0 - (pixels / 255.0)

    return pixels.reshape(1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 1)
