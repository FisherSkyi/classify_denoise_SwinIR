import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from pathlib import Path

def predict_images(directory_path: str, model_path: str, output_csv: str) -> None:
    """
    Classify images in all subdirectories of the given directory using the specified model
    and save results to a CSV file.
    
    Args:
        directory_path (str): Path to the directory containing subfolders with images.
        model_path (str): Path to the .keras model file.
        output_csv (str): Path to the output CSV file for storing results.
    """
    # Validate inputs
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: Directory '{directory_path}' does not exist or is not a directory.")
        return
    
    if not Path(model_path).is_file() or not model_path.endswith('.keras'):
        print(f"Error: Model file '{model_path}' does not exist or is not a .keras file.")
        return

    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        return

    # Class labels as specified
    class_labels = ['003', '005', '007', '011', '016', '028', '030', '035', '054', '055']

    # Initialize lists for storing results
    results = []
    total = 0
    correct = 0
    sum_confidence = 0

    # Supported image extensions
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif'}

    # Iterate through all subdirectories
    for subdir in directory.iterdir():
        if subdir.name == '.DS_Store' or not subdir.is_dir():
            continue

        print(f"Processing subdirectory: {subdir.name}")

        # Iterate through all image files in the subdirectory
        for file in subdir.iterdir():
            if file.suffix.lower() not in exts or not file.is_file():
                continue

            try:
                # Load and preprocess the image
                image = load_img(file, target_size=(240, 240))
                image_array = img_to_array(image)
                image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

                # Make predictions
                predictions = model.predict(image_array, verbose=0)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class_label = class_labels[predicted_class_index]

                # Calculate confidence using softmax
                softmax_output = tf.nn.softmax(predictions)
                confidence = softmax_output.numpy()[0][predicted_class_index]

                # Determine actual class from subdirectory name
                actual_class = subdir.name

                # Check if prediction is correct
                is_correct = predicted_class_label == actual_class

                # Store results
                results.append({
                    'Image': file.name,
                    'Predicted Class': predicted_class_label,
                    'Actual Class': actual_class,
                    'Confidence': confidence,
                    'Correct': is_correct
                })

                total += 1
                sum_confidence += confidence
                if is_correct:
                    correct += 1

                print(f"Image: {file.name} | Predicted: {predicted_class_label} | Actual: {actual_class} | Confidence: {confidence:.4f}")

            except Exception as e:
                print(f"  ! Skipped {file.name}: {e}")

    # Calculate average confidence
    avg_confidence = sum_confidence / total if total > 0 else 0

    # Add summary to results
    results.append({
        'Image': 'Summary',
        'Predicted Class': '',
        'Actual Class': '',
        'Confidence': avg_confidence,
        'Correct': f"{correct}/{total} ({correct/total*100:.2f}%)" if total > 0 else 'N/A'
    })

    # Save results to CSV
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
        print(f"Total images: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Average confidence: {avg_confidence:.4f}")
    except Exception as e:
        print(f"Error saving CSV file '{output_csv}': {e}")

def main() -> None:
    # Get user inputs
    model_path = input("Enter the path to the .keras model file: ").strip()
    directory_path = input("Enter the directory path containing images: ").strip()
    output_csv = input("Enter the output CSV file name (e.g., results.csv): ").strip()

    # Ensure output CSV has .csv extension
    if not output_csv.endswith('.csv'):
        output_csv += '.csv'

    predict_images(directory_path, model_path, output_csv)

if __name__ == "__main__":
    main()
