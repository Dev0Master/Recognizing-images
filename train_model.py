"""
MNIST Handwritten Digit Recognition - Training Script
This script loads MNIST data from IDX files, builds a simple neural network,
trains it, and saves the model for later use.
"""

import numpy as np
import struct
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def load_mnist_images(filename):
    """Load MNIST images from IDX3-ubyte file format."""
    with open(filename, 'rb') as f:
        # Read the magic number and dimensions
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

        # Read all image data
        images = np.fromfile(f, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)

        # Normalize pixel values to 0-1 range
        images = images.astype(np.float32) / 255.0

    return images


def load_mnist_labels(filename):
    """Load MNIST labels from IDX1-ubyte file format."""
    with open(filename, 'rb') as f:
        # Read the magic number and number of labels
        magic, num_labels = struct.unpack('>II', f.read(8))

        # Read all label data
        labels = np.fromfile(f, dtype=np.uint8)

    return labels


def build_model():
    """Build a simple neural network for digit classification."""
    model = keras.Sequential([
        # Flatten 28x28 images to 784-dimensional vectors
        layers.Flatten(input_shape=(28, 28)),

        # Hidden layer with 128 neurons and ReLU activation
        layers.Dense(128, activation='relu'),

        # Dropout for regularization (prevents overfitting)
        layers.Dropout(0.2),

        # Output layer with 10 neurons (one per digit 0-9)
        layers.Dense(10, activation='softmax')
    ])

    return model


def main():
    print("=" * 60)
    print("MNIST Handwritten Digit Recognition - Training")
    print("=" * 60)

    # Load training data
    print("\n[1/6] Loading training images...")
    train_images = load_mnist_images('train-images.idx3-ubyte')
    print(f"      Loaded {len(train_images)} training images")

    print("[2/6] Loading training labels...")
    train_labels = load_mnist_labels('train-labels.idx1-ubyte')
    print(f"      Loaded {len(train_labels)} training labels")

    # Load test data
    print("[3/6] Loading test images...")
    test_images = load_mnist_images('t10k-images.idx3-ubyte')
    print(f"      Loaded {len(test_images)} test images")

    print("[4/6] Loading test labels...")
    test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')
    print(f"      Loaded {len(test_labels)} test labels")

    # Build and compile the model
    print("\n[5/6] Building neural network...")
    model = build_model()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Architecture:")
    model.summary()

    # Train the model
    print("\n[6/6] Training model (this may take 2-3 minutes)...")
    print("-" * 60)

    history = model.fit(
        train_images,
        train_labels,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate on test data
    print("\n" + "=" * 60)
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Save the model
    model.save('digit_model.h5')
    print("\n✓ Model saved as 'digit_model.h5'")

    # Visualize some predictions
    print("\nGenerating sample predictions...")
    predictions = model.predict(test_images[:5], verbose=0)

    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(test_images[i], cmap='gray')
        predicted_digit = np.argmax(predictions[i])
        actual_digit = test_labels[i]

        color = 'green' if predicted_digit == actual_digit else 'red'
        plt.title(f'Pred: {predicted_digit}\nTrue: {actual_digit}', color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=100, bbox_inches='tight')
    print("✓ Sample predictions saved as 'sample_predictions.png'")

    print("\n" + "=" * 60)
    print("Training complete! You can now use the model for predictions.")
    print("=" * 60)


if __name__ == '__main__':
    main()
