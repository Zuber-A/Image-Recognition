import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input,decode_predictions
from google.colab import files
from IPython.display import Image
from google.colab.patches import cv2_imshow

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

uploaded = files.upload()

Image('Chimpanzee.jpeg', width = 500)

# Load and preprocess the image
image_path = 'Chimpanzee.jpeg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(image, (224, 224))
preprocessed_image = preprocess_input(resized_image)

# Perform image classification using the pre-trained model
predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
decoded_predictions = decode_predictions(predictions, top=3)[0]

# Loop through each prediction and display the results
for _, label, probability in decoded_predictions:
    # Create a text string with the label and probability
    result_text = f"I see a {label} with {int(probability * 100)}% confidence!"

    # Print the result text
    print(result_text)