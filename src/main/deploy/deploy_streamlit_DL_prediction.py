import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import PIL.Image as Image
import os

# Load your trained model
def load_model():
    # Load the trained Keras model
    model_path = os.path.join(os.path.dirname(__file__), 'model_20241117.keras')

    # Ensure the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Register custom objects
    custom_objects = {
        # 'RandomRotation': tf.keras.layers.RandomRotation,
        'RandomFlip': tf.keras.layers.RandomFlip,
        'RandomZoom': tf.keras.layers.RandomZoom,
        'RandomContrast': tf.keras.layers.RandomContrast
    }

    # Load the trained model with custom objects
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

# Define preprocessing pipeline (same as during training)
IMAGE_SIZE = 256
image_preprocessing = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE, interpolation='bilinear', crop_to_aspect_ratio=False),
    layers.Rescaling(1. / 255),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2)
])

# Correct class labels for your model
label_classes = ['aneurysm', 'cancer', 'tumor']

# Function to process uploaded image
def preprocess_image(img):
    # Resize image
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert to RGB
    img = img.convert('RGB')

    # Convert to numpy array
    img_array = np.array(img)

    # Rescale to [0, 1]
    img_array = img_array / 255.0

    # Apply Random Flip, Rotation, Zoom, Contrast, etc., manually if needed
    # Example: Random Flip
    if np.random.rand() > 0.5:
        img_array = np.flipud(img_array)  # Random vertical flip

    # Expand dimensions to match model input shape (batch size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Main function to organize the Streamlit app
def main():
    st.title("CT Brain Scan Classification")

    # Load the model
    model = load_model()

    # Upload an image
    uploaded_image = st.file_uploader("Choose a CT Scan Image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open and process the image
        img = Image.open(uploaded_image)
        img_array = preprocess_image(img)

        # Display the uploaded image
        st.image(uploaded_image, caption='Uploaded Image', use_container_width=True)

        # Predict with the model
        predictions = model.predict(img_array)

        # Get the predicted class label
        predicted_class = label_classes[np.argmax(predictions)]

        # Display the prediction
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Prediction confidence: {np.max(predictions) * 100:.2f}%")

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()