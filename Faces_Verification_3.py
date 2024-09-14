import cv2
from deepface import DeepFace
from glob import glob
import os
import numpy as np

# Use only the 'dlib' model
model_name = "Dlib"

# Define a distance threshold for verification
distance_threshold = 0.1

# Get all image files in the directory (recursive)
image_paths = glob("face-db/**/*", recursive=True)
# Filter for image files
image_paths = [path for path in image_paths if path.endswith(('.jpeg', '.jpg', '.png'))]

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting video stream...")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform face recognition
    try:
        results = DeepFace.find(img_path=frame, db_path="face-db", model_name=model_name)

        if results:
            for result in results:
                identity = result['identity'].values[0]
                distance = result['distance'].values[0]

                # Print the result
                print(f"Identity: {identity}")
                print(f"Distance: {distance}")

                # Verify if the distance is below the threshold
                is_match = distance < distance_threshold
                print(f"Match: {is_match}")

        # Show the frame
        cv2.imshow("Webcam", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error in face recognition: {e}")

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
