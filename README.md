# SketchGuess AI

SketchGuess AI is a small Streamlit machine learning web app starter. Users can draw a simple doodle on a canvas, get three placeholder guesses, mark the result as correct or wrong, and see a session score.

The app starts in a dummy prediction stage. It preprocesses the canvas image into a model-ready NumPy array, then `src/predict.py` checks whether a trained model exists. If `model/sketch_model.keras` and `model/class_names.json` are available, the app uses the real model. If they are missing, it safely falls back to fixed sample guesses.

The training pipeline uses a small selected subset of the [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset). The full dataset contains 345 drawing categories and millions of examples, but this app uses 10 hand-picked classes for a lightweight showcase demo:

```text
apple, banana, bicycle, car, cat, dog, fish, house, star, tree
```

## Project Structure

```text
sketchguess-ai/
|-- app.py
|-- requirements.txt
|-- README.md
|-- assets/
|-- data/
|-- model/
|-- notebooks/
|-- scripts/
|   |-- download_quickdraw_data.py
|   `-- train_model.py
`-- src/
    |-- config.py
    |-- preprocessing.py
    |-- predict.py
    `-- ui_helpers.py
```

## Run Locally

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the app:

```powershell
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal, usually `http://localhost:8501`.

## Download Dataset

Download the selected Quick, Draw! NumPy bitmap files:

```powershell
python scripts/download_quickdraw_data.py
```

The files are saved in `data/raw/`. If a file already exists, the script skips it.

## Train Model

Train the first small CNN model:

```powershell
python scripts/train_model.py
```

By default, the trainer uses 5,000 samples per class and trains for 5 epochs so the first run stays manageable.

You can make a quicker test run with fewer samples or epochs:

```powershell
python scripts/train_model.py --samples-per-class 1000 --epochs 2
```

After training, the script saves:

```text
model/sketch_model.keras
model/class_names.json
```

Run the Streamlit app again after training, and it will use the saved model automatically. If those files are missing, the app keeps using dummy predictions.
