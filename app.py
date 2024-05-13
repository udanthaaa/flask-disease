from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app, origins=['http://10.0.2.2:5000'])  # Allow requests from React Native app

# Load the trained model
loaded_model = tf.keras.models.load_model("LAST_leaf_detection_model.h5")

# Function to process uploaded image and make predictions
def process_image(image_path):
    # Load the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Make prediction
    predictions = loaded_model.predict(image)
    if predictions[0] >= 0.5:
        return 'Leaves'
    else:
        return 'Not Leaves'

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/check', methods=['POST'])
def check():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save the uploaded image to a temporary file
    image_path = 'temp_image.jpg'
    image_file.save(image_path)

    # Process the image
    result = process_image(image_path)

    # Delete the temporary image file
    os.remove(image_path)

    return jsonify({'result': result})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save the uploaded image to a temporary file
    image_path = 'temp_image.jpg'
    image_file.save(image_path)

    # Load the trained model for diseases detection
    loaded_model = tf.keras.models.load_model("Dieasese")

    # Load the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)  # Expand dimensions to create a batch

    # Make prediction
    predictions = loaded_model.predict(image_array)
    predicted_class_index = tf.argmax(predictions[0]).numpy()

    class_names = ['Algal Leaf', 'Anthracnose', 'Bird Eye Spot', 'Brown Blight', 'Gray Light', 'Healthy', 'Red Leaf Spot', 'White Spot']

    predicted_class_name = class_names[predicted_class_index]

    # Delete the temporary image file
    os.remove(image_path)

    return jsonify({'predicted_class': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
