import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


def classify(*, model_name = 'og_model.keras', image_path):
    new_model = tf.keras.models.load_model(model_name) # Replace with user input image 
    image = load_img(image_path, target_size=(240, 240))
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension


    # Make predictions using the loaded model
    predictions = new_model.predict(image_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Get the confidence for the predicted class
    confidence = predictions[0][predicted_class_index] * 100  # Convert to percentage

    # Print the predicted class index and corresponding class label
    print('Predicted Class Index:', predicted_class_index)

    # Assuming you have a list of class labels
    class_labels = ['003','005','007','011','016','028','030','035','054','055']  # Replace with your class labels
    predicted_class_label = class_labels[predicted_class_index]
    print('Predicted Class Label:', predicted_class_label)
    print('Prediction Confidence (%):', confidence)

    # Get softmax values for all categories
    softmax_output = tf.nn.softmax(predictions)

    # Convert softmax output to a numpy array for easier manipulation
    softmax_values = softmax_output.numpy()

    # Print softmax values for all categories
    print(softmax_values)
    print(softmax_values[0][predicted_class_index])