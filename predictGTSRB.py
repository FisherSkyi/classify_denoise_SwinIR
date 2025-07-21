import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from pathlib import Path

def predict_images(directory_path: str, model_path: str, gt_csv_path: str, output_csv: str = "results.csv") -> None:
    """
    Classify images in the given directory using the specified model, compare with ground truth from CSV,
    and save results to a CSV file.

    Args:
        directory_path (str): Path to the directory containing images (no subfolders).
        model_path (str): Path to the .keras model file.
        gt_csv_path (str): Path to the ground truth CSV file with Filename and ClassId columns.
        output_csv (str): Path to the output CSV file for storing results (default: results.csv).
    """
    # Validate inputs
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: Directory '{directory_path}' does not exist or is not a directory.")
        return
    
    if not Path(model_path).is_file() or not model_path.endswith('.keras'):
        print(f"Error: Model file '{model_path}' does not exist or is not a .keras file.")
        return
    
    if not Path(gt_csv_path).is_file() or not gt_csv_path.endswith('.csv'):
        print(f"Error: Ground truth CSV file '{gt_csv_path}' does not exist or is not a .csv file.")
        return

    # Load ground truth CSV with semicolon separator
    try:
        gt_df = pd.read_csv(gt_csv_path, sep=';')
        # Verify required columns
        if gt_df.empty or 'Filename' not in gt_df.columns or 'ClassId' not in gt_df.columns:
            print(f"Error: CSV '{gt_csv_path}' must contain 'Filename' and 'ClassId' columns.")
            return
        # Convert Filename to string and ClassId to int for consistency
        gt_df['Filename'] = gt_df['Filename'].astype(str)
        gt_df['ClassId'] = gt_df['ClassId'].astype(int)
    except Exception as e:
        print(f"Error reading CSV '{gt_csv_path}': {e}")
        return

    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        return

    # Class labels: integers 0 to 42
    class_labels = [str(i) for i in range(43)]  # ['0', '1', ..., '42']

    # Initialize lists for storing results
    results = []
    total = 0
    correct = 0
    sum_confidence = 0

    # Supported image extensions
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif', '.ppm'}

    # Iterate through all image files in the directory
    print(f"Processing directory: {directory_path}")
    for file in directory.iterdir():
        if file.suffix.lower() not in exts or not file.is_file():
            continue

        filename = file.name
        try:
            # Find ground truth ClassId from CSV
            gt_row = gt_df[gt_df['Filename'] == filename]
            if gt_row.empty:
                print(f"  ! Skipped {filename}: Not found in ground truth CSV.")
                continue
            actual_class = str(gt_row['ClassId'].iloc[0])  # Convert to string for comparison

            if actual_class not in class_labels:
                print(f"  ! Skipped {filename}: Invalid ClassId {actual_class} in CSV.")
                continue

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

            # Check if prediction is correct
            is_correct = predicted_class_label == actual_class

            # Store results
            results.append({
                'Filename': filename,
                'Predicted Class': predicted_class_label,
                'Correct': is_correct,
                'Confidence': confidence
            })

            total += 1
            sum_confidence += confidence
            if is_correct:
                correct += 1

            print(f"Image: {filename} | Predicted: {predicted_class_label} | Actual: {actual_class} | Confidence: {confidence:.4f}")

        except Exception as e:
            print(f"  ! Skipped {filename}: {e}")

    # Calculate average confidence and incorrect count
    avg_confidence = sum_confidence / total if total > 0 else 0
    incorrect = total - correct

    # Add summary to results
    results.append({
        'Filename': 'Summary',
        'Predicted Class': '',
        'Correct': '',
        'Confidence': avg_confidence
    })
    results.append({
        'Filename': 'Total Correct',
        'Predicted Class': '',
        'Correct': correct,
        'Confidence': ''
    })
    results.append({
        'Filename': 'Total Incorrect',
        'Predicted Class': '',
        'Correct': incorrect,
        'Confidence': ''
    })

    # Save results to CSV
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
        print(f"Total images: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Incorrect predictions: {incorrect}")
        print(f"Average confidence: {avg_confidence:.4f}")
    except Exception as e:
        print(f"Error saving CSV file '{output_csv}': {e}")

def main() -> None:
    # Get user inputs
    model_path = input("Enter the path to the .keras model file: ").strip()
    directory_path = input("Enter the directory path containing images: ").strip()
    gt_csv_path = input("Enter the path to the ground truth CSV file: ").strip()

    # Use fixed output CSV name
    output_csv = "results.csv"

    predict_images(directory_path, model_path, gt_csv_path, output_csv)

if __name__ == "__main__":
    main()
