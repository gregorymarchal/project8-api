import os
from flask import Flask, request, jsonify, send_file
import numpy as np
import skimage as ski
import tensorflow as tf
from werkzeug.utils import secure_filename
from io import BytesIO

HEIGHT = 128
WIDTH = 256
DIMENSIONS = (HEIGHT, WIDTH)

CATEGORY_IDS_TO_COLORS = {
    0: (27, 10, 11),
    1: (213, 104, 165),
    2: (140, 118, 124),
    3: (194, 174, 84),
    4: (129, 196, 93),
    5: (70, 130, 180),
    6: (237, 10, 30),
    7: (13, 16, 112)
}

# Create a Flask application instance
app = Flask(__name__)

# Load the pre-trained model for semantic segmentation
model = tf.keras.models.load_model("mobile_net_fpn_model-final-best.keras")

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def one_hot_to_rgb(one_hot_matrix, color_dict=CATEGORY_IDS_TO_COLORS):
    class_labels = np.argmax(one_hot_matrix, axis=-1)
    h, w = class_labels.shape
    rgb_matrix = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in color_dict.items():
        rgb_matrix[class_labels == label] = color

    return rgb_matrix

def perform_segmentation(image_path, dimensions=DIMENSIONS):
    image = ski.io.imread(image_path)
    image = ski.transform.resize(image, dimensions, anti_aliasing=True)
    input_image = np.expand_dims(image, axis=0)  # Add batch dimension

    predicted_output = model.predict(input_image)
    predicted_output = np.squeeze(predicted_output, axis=0)

    rgb_matrix = one_hot_to_rgb(predicted_output, CATEGORY_IDS_TO_COLORS)

    return rgb_matrix

# Root URL: Health Check
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running", "version": "1.0.0"}), 200

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform segmentation on the uploaded file
        rgb_mask = perform_segmentation(file_path)

        # Convert the RGB mask to a PNG image in memory
        img_io = BytesIO()
        ski.io.imsave(img_io, rgb_mask, format='png')
        img_io.seek(0)

        # Send the image back in the response
        return send_file(img_io, mimetype='image/png')

    return jsonify({'error': 'File type not allowed'}), 400
