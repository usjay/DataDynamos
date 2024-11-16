# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Install TensorFlow (if not already installed)
!pip install tensorflow

# Step 3: Load the .h5 model
import tensorflow as tf

# Path to your .h5 model file in Google Drive
h5_model_path = '/content/drive/MyDrive/eye_state_cnn_model.h5'

# Load the model
model = tf.keras.models.load_model(h5_model_path)

# Step 4: Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Step 5: Save the TFLite model
# Path where you want to save the TFLite model in Google Drive
tflite_model_path = '/content/drive/MyDrive/eye_state_cnn_model.tflite'

# Write the TFLite model to the specified path
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("Model converted and saved successfully.")




