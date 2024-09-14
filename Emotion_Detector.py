import deepface as df
from glob import glob
import os
import threading

# Use only the 'dlib' model (optional, adjust as needed)
import deepface.DeepFace

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

def recognize_faces(db_path="face-db", model_name=model_name):
    """
    Function to perform face recognition using DeepFace.stream
    """
    while True:
        try:
            # Use DeepFace.stream to capture and analyze frames
            results = deepface.DeepFace.stream(db_path=db_path, model_name=model_name)

            if results:
                for result in results:
                    image_path = os.path.normpath(result['identity'].values[0])
                    distance = result['distance'].values[0]

                    # Determine the identity
                    subdirectory_name = image_to_subdir.get(image_path, "Unknown")
                    print(f"Identity: {subdirectory_name}")

                    # You can further process the results here,
                    # e.g., drawing rectangle and adding text (optional)

        except Exception as e:
            print(f"Error in face recognition: {e}")

if __name__ == "__main__":
    # Start recognition thread
    recognition_thread = threading.Thread(target=recognize_faces, daemon=True)
    recognition_thread.start()

    recognition_thread.join()  # Wait for the recognition thread to finish

    print("Exiting program...")