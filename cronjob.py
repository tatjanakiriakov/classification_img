# Script to automate batch predictions using Flask app

import os
import requests
from datetime import datetime
from PIL import Image
import numpy as np
import joblib

model = joblib.load('model.joblib')

category_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Directory containing images
image_dir = '/Users/tatjanakiriakov/Documents/Uni/From Model to Production/images_batches/train'

# URL of your Flask app
flask_url = 'http://localhost:5001/predict_batch'  # Update with your actual URL

# Function to send predictions request
def send_predictions_request():
    # List all files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png', 'bmp'))]

    predictions_batch = []

    for image_file in image_files:
        # Open and preprocess the image
        image_path = os.path.join(image_dir, image_file)
        try:
            image = Image.open(image_path)
        except IOError:
            print(f"Skipping file {image_path}: Not a valid image file")
            continue

        # Convert to grayscale if image has multiple channels
        if image.mode != 'L':
            image = image.convert('L')

        # Resize image to match model input shape
        image = image.resize((28, 28))
        image = np.array(image)
        image = image / 255.0  # Scale pixel values to the range [0, 1]

        # Ensure the image has the correct shape
        if image.shape != (28, 28):
            print(f"Skipping file {image_path}: Image has incorrect shape {image.shape}")
            continue

        # Reshape to match model input shape
        image = image.reshape(1, -1)  # Reshape to 1D array

        # Make predictions
        predictions = model.predict_proba(image)[0]

        # Format predictions
        results = [round(prob, 4) for prob in predictions]

        predictions_batch.append(results)

    # Create response JSON
    response = {
        'predictions_batch': predictions_batch,
        'class_names': category_names
    }

    return response

def main():
    try:
        # Send predictions request
        response = send_predictions_request()

        # Process response as needed
        print("Predictions received:", response)

    except Exception as e:
        print("Error occurred:", e)

if __name__ == "__main__":
    main()
