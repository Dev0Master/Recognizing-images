"""
Flask API Server for Digit Recognition
Handles image predictions from the web interface
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Create directory for training data
TRAINING_DATA_DIR = 'user_training_data'
if not os.path.exists(TRAINING_DATA_DIR):
    os.makedirs(TRAINING_DATA_DIR)

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('digit_model.h5')
print("Model loaded successfully!")

# Counter for collected samples
collected_samples = 0


@app.route('/')
def home():
    """Home route - API status"""
    return jsonify({
        'status': 'running',
        'message': 'Digit Recognition API is ready!',
        'endpoints': {
            '/predict': 'POST - Send image data for prediction',
            '/feedback': 'POST - Submit feedback for learning'
        }
    })


def process_image(image_data):
    """Process base64 image data and return numpy array for prediction"""
    # Remove the data URL prefix if present
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    # Decode base64 to image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to grayscale and resize to 28x28
    image = image.convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize
    image_array = np.array(image).astype(np.float32) / 255.0

    # Invert colors (MNIST expects white digits on black background)
    image_array = 1.0 - image_array

    return image_array


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict digit from base64 encoded image
    Expects JSON: { "image": "data:image/png;base64,..." }
    Returns: { "prediction": 5, "confidence": 0.98, "probabilities": [...] }
    """
    try:
        # Get image data from request
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Process image
        image_array = process_image(data['image'])

        # Reshape for model input (1, 28, 28)
        image_array = image_array.reshape(1, 28, 28)

        # Make prediction
        predictions = model.predict(image_array, verbose=0)[0]

        # Get predicted digit and confidence
        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit])

        # Convert all probabilities to regular Python floats
        probabilities = [float(p) for p in predictions]

        # Return results
        return jsonify({
            'prediction': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Handle user feedback and save training data
    Expects JSON: { "image": "data:image/png;base64,...", "label": 5, "was_correct": true }
    """
    global collected_samples

    try:
        data = request.get_json()

        if 'image' not in data or 'label' not in data:
            return jsonify({'error': 'Missing image or label'}), 400

        label = int(data['label'])
        was_correct = data.get('was_correct', False)

        # Process and save the image
        image_array = process_image(data['image'])

        # Save to training data directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        collected_samples += 1

        filename = f"{timestamp}_{collected_samples}_digit{label}.npy"
        filepath = os.path.join(TRAINING_DATA_DIR, filename)

        np.save(filepath, image_array)

        # Save metadata
        metadata = {
            'label': label,
            'was_correct': was_correct,
            'timestamp': timestamp,
            'sample_id': collected_samples
        }

        metadata_file = filepath.replace('.npy', '_meta.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        print(f"Saved training sample: {filename} (Label: {label}, Correct: {was_correct})")

        return jsonify({
            'status': 'success',
            'message': f'Training data saved! Total samples collected: {collected_samples}',
            'samples_collected': collected_samples
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Retrain the model with collected user data
    """
    global model

    try:
        # Check if we have training data
        training_files = [f for f in os.listdir(TRAINING_DATA_DIR) if f.endswith('.npy')]

        if len(training_files) < 5:
            return jsonify({
                'status': 'error',
                'message': f'Need at least 5 samples to retrain. Currently have {len(training_files)}.'
            }), 400

        # Load user training data
        user_images = []
        user_labels = []

        for filename in training_files:
            filepath = os.path.join(TRAINING_DATA_DIR, filename)
            metadata_file = filepath.replace('.npy', '_meta.json')

            image_array = np.load(filepath)

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            user_images.append(image_array)
            user_labels.append(metadata['label'])

        user_images = np.array(user_images)
        user_labels = np.array(user_labels)

        print(f"\nRetraining with {len(user_images)} user samples...")

        # Fine-tune the model
        history = model.fit(
            user_images,
            user_labels,
            epochs=10,
            batch_size=min(32, len(user_images)),
            verbose=1
        )

        # Save the updated model
        model.save('digit_model.h5')

        final_accuracy = history.history['accuracy'][-1]

        print(f"Retraining complete! Final accuracy: {final_accuracy * 100:.2f}%")

        return jsonify({
            'status': 'success',
            'message': f'Model retrained with {len(user_images)} samples!',
            'accuracy': float(final_accuracy),
            'samples_used': len(user_images)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Digit Recognition API Server")
    print("=" * 60)
    print("\nServer starting on http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
