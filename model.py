import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib



def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8)
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

# Preprocess the data
train_images_flat = train_images.reshape(train_images.shape[0], -1) / 255.0
test_images_flat = test_images.reshape(test_images.shape[0], -1) / 255.0

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(train_images_flat, train_labels)

# Evaluate the model
accuracy = model.score(test_images_flat, test_labels)
print("Accuracy:", accuracy)

# Predict on test set
predictions = model.predict(test_images_flat)

# Print results
category_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
for i, (true_label, predicted_label) in enumerate(zip(test_labels, predictions)):
    print(f"Image {i+1}: True Label = {category_names[true_label]}, Predicted Label = {category_names[predicted_label]}")

# Save the model without specifying the .h5 extension
joblib.dump(model, 'model.joblib')
