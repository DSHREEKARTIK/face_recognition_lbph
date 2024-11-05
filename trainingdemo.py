import cv2
import os
import numpy as np

def train_recognizer(dataset_dir):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images = []
    labels = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".jpg"):
                label = int(os.path.basename(os.path.normpath(root)))  # Ensure label is an integer
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(label)

    if not images:
        print("No training data found.")
        return

    recognizer.train(images, np.array(labels))
    recognizer.save("Trainer.yml")
    print("Training Completed.")

if __name__ == "__main__":
    dataset_dir = "datasets"
    train_recognizer(dataset_dir)
