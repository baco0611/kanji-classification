import os
import pandas as pd
import numpy as np
import cv2

# Function to load images from a folder and apply thresholding
def load_images_from_folder(folder, label):
    images = []
    labels = []

    # Check if the given path is a directory
    if os.path.isdir(folder):
        # Loop through all the files in the directory
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            # Read the image in grayscale mode
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Proceed if the image is not None (successfully loaded)
            if img is not None:
                # Resize the image to 28x28 pixels
                img_resized = cv2.resize(img, (28, 28))
                # Apply Otsu's thresholding to binarize the image
                _, optimal_thresh = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                images.append(optimal_thresh)
                labels.append(label)

    return images, labels

dataset_path = "./image/dataset"  # Path to dataset folder
data_path = "./data/data/"        # Path to save data
csv_path = data_path + "label_unicode.csv"  # Path to label CSV file

images_all = []
labels_all = []
df = pd.read_csv(csv_path)  # Load label information from CSV

# Loop through each row of the CSV (each folder and its corresponding label)
for index, row in df.iterrows():
    label = row['Label']
    unicode = row['Unicode']

    # Path to the folder containing the images for this label
    image_folder = os.path.join(dataset_path, unicode)

    # Load images and their corresponding labels
    images, labels = load_images_from_folder(image_folder, label)

    # Extend the main image and label lists
    images_all.extend(images)
    labels_all.extend(labels)

# Print the number of images and labels loaded
print("Number of images:", len(images_all))
print("Number of labels:", len(labels_all))

# Flatten each image (convert from 2D to 1D) for easier storage
flattened_images = [image.flatten() for image in images_all]
flattened_images_np = np.array(flattened_images)
labels_all_np = np.array(labels_all)

# Save the images and labels as a .npz file
np.savez(data_path + 'training_data.npz', images = flattened_images_np, labels = labels_all_np)
print(f"Saved training data to folder {data_path + 'training_data.npz'}")
