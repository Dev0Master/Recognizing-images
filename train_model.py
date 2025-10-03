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
from feature_extractor import extract_features_batch


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
    """Build a neural network with CNN and geometric feature learning."""
    # Image input branch (CNN)
    image_input = keras.Input(shape=(28, 28, 1), name='image_input')

    # Convolutional layers for pattern recognition
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Geometric features input branch
    feature_input = keras.Input(shape=(15,), name='feature_input')
    f = layers.Dense(32, activation='relu')(feature_input)
    f = layers.Dropout(0.2)(f)

    # Combine both branches
    combined = layers.concatenate([x, f])
    combined = layers.Dense(64, activation='relu')(combined)
    combined = layers.Dropout(0.2)(combined)

    # Output layer
    output = layers.Dense(10, activation='softmax', name='output')(combined)

    # Create model with two inputs
    model = keras.Model(inputs=[image_input, feature_input], outputs=output)

    return model


def main():
    print("=" * 60)
    print("MNIST Handwritten Digit Recognition - Training")
    print("=" * 60)

    # Load training data
    print("\n[1/7] Loading training images...")
    train_images = load_mnist_images('train-images.idx3-ubyte')
    print(f"      Loaded {len(train_images)} training images")

    print("[2/7] Loading training labels...")
    train_labels = load_mnist_labels('train-labels.idx1-ubyte')
    print(f"      Loaded {len(train_labels)} training labels")

    # Load test data
    print("[3/7] Loading test images...")
    test_images = load_mnist_images('t10k-images.idx3-ubyte')
    print(f"      Loaded {len(test_images)} test images")

    print("[4/7] Loading test labels...")
    test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')
    print(f"      Loaded {len(test_labels)} test labels")

    # Extract geometric features
    print("\n[5/7] Extracting geometric features from training data...")
    print("      (circles, triangles, lines, loops, symmetry, etc.)")
    train_features = extract_features_batch(train_images)
    print(f"      Extracted {train_features.shape[1]} features per image")

    print("[6/7] Extracting geometric features from test data...")
    test_features = extract_features_batch(test_images)
    print(f"      Extracted {test_features.shape[1]} features per image")

    # Build and compile the model
    print("\n[7/7] Building enhanced neural network...")
    print("      CNN branch + Geometric feature branch")
    model = build_model()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Architecture:")
    model.summary()

    # Prepare data for dual-input model
    train_images_cnn = train_images.reshape(-1, 28, 28, 1)
    test_images_cnn = test_images.reshape(-1, 28, 28, 1)

    # Train the model
    print("\nTraining model with self-learning features...")
    print("-" * 60)

    history = model.fit(
        [train_images_cnn, train_features],
        train_labels,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate on test data
    print("\n" + "=" * 60)
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(
        [test_images_cnn, test_features],
        test_labels,
        verbose=0
    )

    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Save the model
    model.save('digit_model.h5')
    print("\n✓ Model saved as 'digit_model.h5'")

    # Visualize some predictions
    print("\nGenerating sample predictions...")
    predictions = model.predict(
        [test_images_cnn[:5], test_features[:5]],
        verbose=0
    )

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
