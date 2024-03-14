from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import os
import joblib

app = Flask(__name__)

model = joblib.load('model.joblib')
category_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    images_dir = '/Users/tatjanakiriakov/Documents/Uni/From Model to Production/images_batches/train'

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('jpg', 'jpeg', 'png', 'bmp'))]

    predictions_batch = []

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        try:
            image = Image.open(image_path)
        except IOError:
            print(f"Skipping file {image_path}: Not a valid image file")
            continue

        if image.mode != 'L':
            image = image.convert('L')

        image = image.resize((28, 28))
        image = np.array(image)
        image = image / 255.0

        if image.shape != (28, 28):
            print(f"Skipping file {image_path}: "
                  f"Image has incorrect shape {image.shape}")
            continue

        image = image.reshape(1, -1)


        predictions = model.predict_proba(image)[0]
        results = [round(prob, 4) for prob in predictions]
        predictions_batch.append(results)

    response = {
        'predictions_batch': predictions_batch,
        'class_names': category_names
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
