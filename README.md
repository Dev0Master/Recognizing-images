# Handwritten Digit Recognition System

A complete machine learning application that recognizes handwritten digits (0-9) using a neural network trained on the MNIST dataset.

## Project Overview

This project includes:
- **Neural Network Model**: Simple but effective architecture with 97.29% accuracy
- **Flask API Server**: Backend for handling predictions
- **Interactive Web Interface**: HTML/CSS/JavaScript canvas for drawing digits

## Files in This Project

- `train_model.py` - Script to train the neural network
- `digit_model.h5` - Trained model (generated after training)
- `predict.py` - Flask API server for predictions
- `index.html` - Main web interface
- `style.css` - Styling for the web interface
- `script.js` - Drawing and prediction logic
- `train-images.idx3-ubyte` - MNIST training images (60,000 samples)
- `train-labels.idx1-ubyte` - MNIST training labels
- `t10k-images.idx3-ubyte` - MNIST test images (10,000 samples)
- `t10k-labels.idx1-ubyte` - MNIST test labels

## How to Use

### Step 1: Start the Flask Server

Open a terminal/command prompt and run:

```bash
cd "C:\Users\AL-nabaa\OneDrive\Desktop\Recognizing images"
python predict.py
```

You should see:
```
Loading model...
Model loaded successfully!

============================================================
Digit Recognition API Server
============================================================

Server starting on http://localhost:5000
Press Ctrl+C to stop the server
```

**Keep this terminal window open!** The server must be running for the web interface to work.

### Step 2: Open the Web Interface

Simply open `index.html` in your web browser:
- Double-click on `index.html`, or
- Right-click → Open with → Your preferred browser

### Step 3: Draw and Predict!

1. **Draw a digit** (0-9) on the white canvas using your mouse
2. Try to draw in the center and make it reasonably large
3. Click **"Predict Digit"** to see what the AI thinks you drew
4. View the prediction and probability distribution for all digits
5. Click **"Clear Canvas"** to try another digit

## Model Details

### Architecture
```
Input Layer:    784 neurons (28x28 pixels flattened)
Hidden Layer:   128 neurons with ReLU activation
Dropout:        20% (prevents overfitting)
Output Layer:   10 neurons (digits 0-9) with softmax
```

### Training Results
- **Test Accuracy**: 97.29%
- **Training Time**: ~5 epochs (~2-3 minutes on CPU)
- **Total Parameters**: 101,770

## Retraining the Model

If you want to retrain the model from scratch:

```bash
python train_model.py
```

This will:
1. Load the MNIST dataset
2. Build the neural network
3. Train for 5 epochs
4. Evaluate on test data
5. Save the model as `digit_model.h5`

## Troubleshooting

### "Connection refused" or API errors
- Make sure the Flask server is running (`python predict.py`)
- Check that the server is on `http://localhost:5000`

### Poor predictions
- Try drawing larger, clearer digits
- Draw in the center of the canvas
- Make sure lines are thick and clear
- The model was trained on handwritten digits, so print-style digits may not work as well

### Module not found errors
- Reinstall dependencies: `pip install tensorflow flask flask-cors numpy pillow`

## System Requirements

- **Python**: 3.13+ (you have 3.13.7)
- **RAM**: 2GB+ recommended
- **Storage**: ~500MB for libraries and model
- **GPU**: Optional (CPU training is sufficient for this project)

## Technologies Used

- **Machine Learning**: TensorFlow/Keras
- **Backend**: Flask (Python web framework)
- **Frontend**: HTML5, CSS3, JavaScript
- **Dataset**: MNIST (Modified National Institute of Standards and Technology)

## Future Enhancements

Possible improvements you could add:
- Support for multi-digit numbers
- Save/load drawings
- Training on custom handwriting samples
- Mobile-friendly interface
- Real-time prediction while drawing
- Export predictions to CSV

---

**Enjoy experimenting with handwritten digit recognition!**
