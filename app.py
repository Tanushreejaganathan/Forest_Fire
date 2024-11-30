from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Set paths for the model and uploads directory
MODEL_PATH = Path('models/forest_fire_cnn.keras')
UPLOAD_FOLDER = Path('static/uploads')

# Ensure the uploads directory exists
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Load the trained model
model = load_model(MODEL_PATH.as_posix())

# Image dimensions
img_width, img_height = 224, 224

# Define home route
@app.route('/')
def index():
    return render_template('index.html')

# Define route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file
        file_path = UPLOAD_FOLDER / file.filename
        file.save(file_path)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(img_width, img_height))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale as done during training

        # Make prediction
        prediction = model.predict(img_array)
        result = " No Fire" if prediction >= 0.5 else "Fire"

        # Render the result page
        return render_template('result.html', prediction=result, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
