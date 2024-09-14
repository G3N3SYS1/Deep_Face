import cv2
from deepface import DeepFace
from glob import glob
import os
import time
import sys
import io
import threading
import queue


# Function to suppress stdout
class SuppressOutput(io.StringIO):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout


# Use only the 'dlib' model
model_name = "Dlib"

# Define a distance threshold for verification
distance_threshold = 0.1

# Initialize the image_to_subdir dictionary
image_to_subdir = {}

# Check directory and file reading
print("Populating image_to_subdir dictionary...")

# Get all subdirectories
subdirs = glob("face-db/*")
print(f"Subdirectories found: {subdirs}")

for subdir in subdirs:
    if os.path.isdir(subdir):
        print(f"Processing subdirectory: {subdir}")
        # Get all image files in the subdirectory
        images = glob(os.path.join(subdir, "*.jpeg")) + \
                 glob(os.path.join(subdir, "*.jpg")) + \
                 glob(os.path.join(subdir, "*.png"))

        for image_path in images:
            if image_path.endswith(('.jpeg', '.jpg', '.png')):
                # Normalize paths
                image_path = os.path.normpath(image_path)
                subdir_name = os.path.basename(subdir)
                image_to_subdir[image_path] = subdir_name

# Print out the dictionary for debugging
print("Image to Subdirectory Mapping:")
for image_path, subdir in image_to_subdir.items():
    print(f"{image_path} -> {subdir}")

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a queue to hold frames
frame_queue = queue.Queue()

def capture_frames():
    global cap, frame_queue
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        if not frame_queue.full():
            frame_queue.put(frame)

def recognize_faces():
    global frame_queue
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            start_time = time.time()
            try:
                # Suppress the output from DeepFace
                with SuppressOutput():
                    results = DeepFace.find(img_path=frame, db_path="face-db", model_name=model_name)
                end_time = time.time()

                if results:
                    for result in results:
                        image_path = os.path.normpath(result['identity'].values[0])
                        distance = result['distance'].values[0]

                        # Determine the identity
                        subdirectory_name = image_to_subdir.get(image_path, "Unknown")

                        # Print only the desired information
                        print(f"Identity: {subdirectory_name}")
                        #print(f"Distance: {distance}")
                        #print(f"Match: {distance < distance_threshold}")

            except Exception as e:
                print(f"Error in face recognition: {e}")

def main():
    # Start capture and recognition threads
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    recognition_thread = threading.Thread(target=recognize_faces, daemon=True)

    capture_thread.start()
    recognition_thread.start()

    while True:
        # Display the frame from the queue
        if not frame_queue.empty():
            frame = frame_queue.get()
            cv2.imshow("Webcam", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
