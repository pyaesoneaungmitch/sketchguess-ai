# SketchGuess AI: Project Summary

## What It Is

SketchGuess AI is a small end-to-end machine learning project that turns hand-drawn doodles into live object predictions. A user draws on a Streamlit canvas, and a TensorFlow/Keras convolutional neural network returns the three most likely objects with confidence scores.

The current model recognizes 10 categories: apple, banana, bicycle, car, cat, dog, fish, house, star, and tree.

## Why I Built It

I built this project to move beyond training a model in isolation and show the complete path from raw public data to an interactive application. Doodle recognition makes image classification easy to understand visually, while still requiring meaningful decisions about preprocessing, model design, evaluation, and user experience.

## What Users Can Do

- Choose a supported object and draw it on a browser-based canvas
- Request the model's top three predictions
- Compare confidence scores visually
- Mark the result Correct or Wrong
- Track attempts and accuracy during the current session
- View whether the app is using the trained model or its safe dummy fallback
- Inspect saved training accuracy, loss, and model details

## ML Concepts Demonstrated

- Multi-class image classification
- Grayscale image preprocessing and normalization
- Cropping and resizing drawings to `28x28`
- Train/test splitting
- Convolutional neural networks with TensorFlow/Keras
- Softmax probabilities and top-k prediction ranking
- Evaluation with accuracy, loss, learning curves, and a confusion matrix
- Reproducible data download, training, and testing scripts

## What I Learned

This project reinforced that a useful ML demo needs more than a good test score. The input shown to the model must match the training representation, model artifacts and class labels must stay aligned, and the interface must explain its supported categories and fallback state clearly. I also gained practical experience structuring a Python project, managing generated files, and presenting model results for non-technical visitors.

## Limitations

The model is intentionally lightweight and supports only 10 Quick, Draw! categories. Real users may draw with styles that differ from the dataset, and visually similar or incomplete sketches can lead to uncertain predictions. The confidence values are not calibrated guarantees, session results are not persisted, and the trained model binary must be generated locally because it is not committed to Git.

Despite those limits, SketchGuess AI demonstrates a complete, understandable workflow from dataset download through model training and evaluation to a polished interactive demo.
