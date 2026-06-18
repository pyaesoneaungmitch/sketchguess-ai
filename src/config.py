from pathlib import Path


CLASS_NAMES = [
    "apple",
    "banana",
    "bicycle",
    "car",
    "cat",
    "dog",
    "fish",
    "house",
    "star",
    "tree",
]

CANVAS_SIZE = 400
MODEL_INPUT_SIZE = (28, 28)

TOP_K_PREDICTIONS = 3
CONFIDENCE_DECIMALS = 0
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0

EMPTY_CANVAS_THRESHOLD = 250
CROP_PADDING = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "model"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_PATH = MODEL_DIR / "sketch_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
MODEL_METRICS_PATH = MODEL_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = ASSETS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PATH = ASSETS_DIR / "training_history.png"
SAMPLE_PREDICTIONS_PATH = ASSETS_DIR / "sample_predictions.png"

QUICKDRAW_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

TRAINING_SAMPLES_PER_CLASS = 5000
TRAINING_EPOCHS = 5
TEST_SPLIT = 0.2
RANDOM_SEED = 42
