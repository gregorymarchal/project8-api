# Project 8: Image Segmentation Flask API

This repository contains a Flask-based API for performing semantic segmentation on images using a pre-trained TensorFlow model.

## Features

- **Upload Images**: Upload an image file in formats like PNG, JPG, JPEG, or GIF.
- **Image Segmentation**: The API processes the uploaded image, performs semantic segmentation using a pre-trained MobileNet-FPN model, and returns the segmented image in RGB format.
- **Health Check**: Basic endpoint to check if the API is running.

## Model

The API uses a pre-trained TensorFlow model (`mobile_net_fpn_model-final-best.keras`) for performing semantic segmentation. The model is loaded when the application starts.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/gregorymarchal/project8-api.git
    cd project8-api
    ```

2. **Create a virtual environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Add your TensorFlow model**:
    
    Place my pre-trained TensorFlow model file https://www.gregorymarchal.com/pub/mobile_net_fpn_model-final-best.keras in the root directory of the project. Or use yours!

## Usage

1. **Run the Flask app**:

    ```bash
    gunicorn -w 4 -b 127.0.0.1:8000 app:app
    # Or 0.0.0.0:8000 if you want it to listen to external IPs.
    ```

2. **Upload an image**:

    You can use `curl` or any HTTP client to upload an image file for segmentation:

    ```bash
    curl -X POST -F 'file=@path_to_your_image.png' http://127.0.0.1:8000/upload --output segmented_image.png
    ```

    The response will be a segmented image in PNG format.

3. **Health Check**:

    Check if the API is running by accessing:

    ```bash
    http://127.0.0.1:8000/
    ```

## Folder Structure

- **app.py**: The main Flask application file.
- **uploads/**: Directory where uploaded files are temporarily stored.
- **requirements.txt**: Python dependencies.
- **README.md**: Documentation file.

## Requirements

- Python 3.10
- TensorFlow 2.15.1
- Flask.
- scikit-image.
- NumPy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
