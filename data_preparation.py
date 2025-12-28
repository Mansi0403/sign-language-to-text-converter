import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATASET_PATH =DATASET_PATH = r"C:\Users\Mansi\OneDrive\Desktop\AI-Assistant-converting-sign-language-into-text-main\asl_alphabet_test"

 

def load_images(dataset_path):
    images = []
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            # Get label from filename: "A_test.jpg" → label "A"
            label_name = filename.split('_')[0]  # Gets "A" from "A_test.jpg"
            
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = np.expand_dims(image, axis=-1)  # Add channel dimension


            if image is not None:
                image = cv2.resize(image, (64, 64))
                images.append(image)
                labels.append(label_name)
    
    print(f"Loaded {len(images)} images")
    return np.array(images), np.array(labels)


images, labels = load_images(DATASET_PATH)

# Load images and labels
images, labels = load_images(DATASET_PATH)

# Normalize images
images = images / 255.0  # Normalize pixel values to [0, 1]

# Get unique labels
unique_labels = np.unique(labels)

# Convert labels to numerical representation
labels_numerical = np.array([np.where(unique_labels == label)[0][0] for label in labels])

# Convert labels to categorical
labels_categorical = to_categorical(labels_numerical)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Optionally, save preprocessed data
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)