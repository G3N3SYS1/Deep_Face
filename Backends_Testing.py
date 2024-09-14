# Backends_Testing.py
# Uses different backends to generate the extracted faces (compare detected face using different backends)

from deepface import DeepFace
import matplotlib.pyplot as plt

# List of available backends for face detection
backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]

# pulls out results of faces using different backends
fig, axs = plt.subplots(3, 2, figsize=(15,10))
axs = axs.flatten()

img_path = "face-db/Elon/Elon-1.jpeg"

for i, b in enumerate(backends):
    try:
        # Extract faces using different backends
        faces = DeepFace.extract_faces(img_path=img_path, detector_backend=b)

        # Check if any face was detected
        if len(faces) > 0:
            face = faces[0]["face"]  # Extract the first detected face
            axs[i].imshow(face)
            axs[i].set_title(f"Backend: {b}")
        else:
            axs[i].set_title(f"No face detected using {b}")

        axs[i].axis("off")
    except Exception as e:
        axs[i].set_title(f"Error with {b}")
        axs[i].axis("off")

plt.tight_layout()
plt.show()
