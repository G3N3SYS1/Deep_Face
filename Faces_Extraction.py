# Faces_Extraction.py
# Opens the image of extracted faces (detected face)

from deepface import DeepFace
import matplotlib.pyplot as plt

# List of available backends for face detection
backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]

# Call DeepFace.extract_faces without target_size argument
faces = DeepFace.extract_faces(
    img_path="face-db/Elon/Elon-1.jpeg", detector_backend="opencv"
)

# Check if any face was detected
if len(faces) > 0:
    face = faces[0]["face"]  # Extract the first detected face
    plt.imshow(face)
    plt.show()
else:
    print("No faces detected.")

