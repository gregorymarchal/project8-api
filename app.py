import os
from flask import Flask, request, jsonify
import skimage as ski
import tensorflow as tf

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
    """
    Converts a one-hot encoded 3D matrix to a 2D matrix of class labels,
    then uses a color dictionary to map class labels to RGB values.

    Parameters:
    one_hot_matrix (np.ndarray): One-hot encoded 3D matrix.
    color_dict (dict): Dictionary mapping class labels to RGB tuples.

    Returns:
    np.ndarray: 3D RGB matrix.
    """
    # Step 1: Convert one-hot encoded matrix to 2D matrix of class labels
    class_labels = np.argmax(one_hot_matrix, axis=-1)

    # Step 2: Create an empty RGB matrix
    h, w = class_labels.shape
    rgb_matrix = np.zeros((h, w, 3), dtype=np.uint8)

    # Step 3: Map class labels to RGB values using the color dictionary
    for label, color in color_dict.items():
        rgb_matrix[class_labels == label] = color

    return rgb_matrix

@app.route("/upload", methods=["POST"])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # If user does not select file, browser may submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'message': f'File successfully uploaded to {file_path}'}), 200

    return jsonify({'error': 'File type not allowed'}), 400

def perform_segmentation(image_path, dimensions=DIMENSIONS):
    # Load the pre-trained TensorFlow model
    model = tf.keras.models.load_model("mobile_net_fpn_model-final-best.keras")

    # Preprocess the image
    image = ski.io.imread(image_path)
    image = ski.transform.resize(image, dimensions)
    input_image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction
    predicted_output = model.predict(input_image)
    predicted_output = np.squeeze(predicted_output, axis=0)

    # Convert one-hot encoded matrix to RGB matrix
    rgb_matrix = one_hot_to_rgb(predicted_output, CATEGORY_IDS_TO_COLORS)

    annotation_path = image_path.replace(".png", "-color.png")

    # Save the predicted output as an image file
    ski.io.imsave(annotation_path, rgb_matrix)

    return annotation_path
