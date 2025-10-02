# Quick Start Guide

## Your Handwritten Digit Recognition System is Ready!

### What's Been Done

✓ **Trained Neural Network** - 97.29% accuracy on MNIST dataset
✓ **Flask API Server** - Currently running on http://localhost:5000
✓ **Web Interface** - Ready to use at [index.html](index.html)

---

## How to Use Right Now

### The Flask server is already running in the background!

**Simply open `index.html` in your browser:**
1. Double-click on `index.html` in this folder
2. Or right-click → Open with → Chrome/Firefox/Edge

### Then:
1. **Draw** a digit (0-9) on the canvas
2. **Click** "Predict Digit"
3. **See** the AI's prediction and confidence scores!

---

## Important Notes

### Server Management

The Flask server is running in the background. To stop it:
- Go back to your terminal/command prompt
- Press `Ctrl+C`

To restart the server later:
```bash
cd "C:\Users\AL-nabaa\OneDrive\Desktop\Recognizing images"
python predict.py
```

### Drawing Tips

For best results:
- Draw **large** digits that fill most of the canvas
- Draw in the **center** of the canvas
- Use **thick, clear** lines
- Draw digits similar to how you'd write them by hand

---

## Files Created

| File | Purpose |
|------|---------|
| `train_model.py` | Training script (already run) |
| `digit_model.h5` | Trained model (97.29% accuracy) |
| `predict.py` | Flask API server (currently running) |
| `index.html` | Web interface |
| `style.css` | Styling |
| `script.js` | Drawing and prediction logic |
| `README.md` | Full documentation |

---

## Next Steps

Want to experiment more? Try:
- Drawing different styles of digits
- Testing edge cases (very small/large digits)
- Drawing digits off-center to see how it affects accuracy
- Retraining the model with different parameters

---

**Have fun testing your digit recognition system!**
