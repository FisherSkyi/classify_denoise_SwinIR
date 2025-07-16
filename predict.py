import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

new_model = tf.keras.models.load_model('my_model.keras')

# Directory containing subdirectories with images
noise_type = input("noise type: ")
# var = input("folder name: ")


for var in range (1,11):
    directory_path = f'files/{noise_type}{var}'  # Replace with the actual directory path
    
    # Output file for saving prediction results
    output_file = f'prediction_results_{noise_type}{var}.txt'
    
    # Assuming you have a list of class labels
    class_labels = ['007','011','012','028','030','035','050','054','055','056'] 
    
    total = 0;
    correct = 0;
    sum_confidence = 0
    avg_confidence = 0
    
    # Iterate through all subdirectories in the directory
    for subdir in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, subdir)
        print(subdirectory_path)
        if '.DS_Store' in subdirectory_path:
            print("skip")
            continue
        
        # Iterate through all image files in the subdirectory
        for filename in os.listdir(subdirectory_path):
            image_path = os.path.join(subdirectory_path, filename)
            
            # Load and preprocess the image for prediction
            image = load_img(image_path, target_size=(240, 240))
            image_array = img_to_array(image)
            image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            
            # Make predictions using the loaded model
            predictions = new_model.predict(image_array)
            
            # Get the predicted class index
            predicted_class_index = np.argmax(predictions[0])
            
            # Get the predicted class label
            predicted_class_label = class_labels[predicted_class_index]
            
            # Print and save the prediction results
            prediction_result = f"Image: {filename} | Predicted Class: {predicted_class_label} | Actual Class: {subdirectory_path} "

            # Get softmax values for all categories
            softmax_output = tf.nn.softmax(predictions)
            
            # Convert softmax output to a numpy array for easier manipulation
            softmax_values = softmax_output.numpy()
            
            # Print softmax values for all categories
            confidence = softmax_values[0][predicted_class_index]
            print(confidence)
            
            print(prediction_result)
            total = total+1
            sum_confidence = sum_confidence + confidence
            if predicted_class_label in subdirectory_path:
                correct = correct+1
            
            with open(output_file, 'a') as file:
                file.write(prediction_result)
                file.write(f'confidence: {confidence}\n')

    avg_confidence = sum_confidence/total
    with open(output_file, 'a') as file:
        file.write(f"total is: {total}\n")
        file.write(f"correct is: {correct}\n")
        file.write(f"average confidence is: {avg_confidence}")
    
