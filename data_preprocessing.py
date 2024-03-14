import numpy as np
from sklearn.preprocessing import OneHotEncoder
import gzip
import os

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Skip the header bytes (magic number and size)
        f.read(16)
        # Read the image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape into images (60000 images, each 28x28 pixels)
        data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Skip the header bytes (magic number and size)
        f.read(8)
        # Read the label data
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

# Define file paths
train_images_file = '/Users/tatjanakiriakov/Documents/Uni/From Model to Production/mist_images/train-images-idx3-ubyte'
train_labels_file = '/Users/tatjanakiriakov/Documents/Uni/From Model to Production/mist_images/train-labels-idx1-ubyte'
test_images_file = '/Users/tatjanakiriakov/Documents/Uni/From Model to Production/mist_images/t10k-images-idx3-ubyte'
test_labels_file = '/Users/tatjanakiriakov/Documents/Uni/From Model to Production/mist_images/t10k-labels-idx1-ubyte'

# Load training and test data
train_images = load_mnist_images(train_images_file)
train_labels = load_mnist_labels(train_labels_file)
test_images = load_mnist_images(test_images_file)
test_labels = load_mnist_labels(test_labels_file)

# Print dataset shapes
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)


# Step 1: Reshape the images
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# Step 2: Normalize the pixel values
train_images_normalized = train_images_flat / 255.0
test_images_normalized = test_images_flat / 255.0

# Step 3: Encode the labels
one_hot_encoder = OneHotEncoder()
train_labels_encoded = one_hot_encoder.fit_transform(train_labels.reshape(-1, 1)).toarray()
test_labels_encoded = one_hot_encoder.transform(test_labels.reshape(-1, 1)).toarray()

# Print shapes of preprocessed data
print("Train images shape:", train_images_normalized.shape)
print("Train labels shape:", train_labels_encoded.shape)
print("Test images shape:", test_images_normalized.shape)
print("Test labels shape:", test_labels_encoded.shape)

# Save preprocessed data
np.savez('/Users/tatjanakiriakov/Documents/Uni/From Model to Production/mist_images/preprocessed_data.npz',
         train_images=train_images_normalized, train_labels=train_labels_encoded,
         test_images=test_images_normalized, test_labels=test_labels_encoded)
