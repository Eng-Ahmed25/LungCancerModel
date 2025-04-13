import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import io
import uvicorn
from uvicorn import Server, Config

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Initialize FastAPI (define app at the top level)
app = FastAPI()

# Load the trained model
model = load_model("cancer.h5")  # Replace with your model path

# Compile the model manually
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Define the ImageDataGenerator with the same preprocessing as during training
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Define class labels
class_labels = ["squamous_cell_carcinoma", "adenocarcinoma", "benign"]

# Define image size and batch size
img_size = (224, 224)
batch_size = 1  # Process one image at a time

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")  # Ensure RGB format

    # Save the image temporarily (required for flow_from_dataframe)
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)

    # Create a temporary DataFrame for the input image
    temp_df = pd.DataFrame({
        "filepaths": [temp_image_path],  # Column for image paths
        "labels": ["unknown"]            # Dummy label (not used for prediction)
    })

    # Use flow_from_dataframe to preprocess the image
    generator = datagen.flow_from_dataframe(
        dataframe=temp_df,
        x_col="filepaths",               # Column with image paths
        y_col="labels",                  # Column with labels (not used for prediction)
        target_size=img_size,            # Same as during training
        class_mode="categorical",        # Same as during training
        color_mode="rgb",                # Same as during training
        shuffle=False,                   # No shuffling for prediction
        batch_size=batch_size            # Process one image at a time
    )

    # Get the preprocessed image
    preprocessed_img = next(generator)

    # Get the model's prediction
    predictions = model.predict(preprocessed_img)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Map the index to the class label
    predicted_class = class_labels[predicted_class_index]

    # Get the prediction probabilities
    prediction_probabilities = {
        class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))
    }

    # Return the prediction as a JSON response
    return {
        "predicted_class": predicted_class,
        "prediction_probabilities": prediction_probabilities
    }

# Run the FastAPI app in a Jupyter notebook or interactive environment
 