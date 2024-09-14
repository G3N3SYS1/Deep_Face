# Faces_Verification.py
# Compares the stated images and returns similarity
from deepface import DeepFace
import matplotlib.pyplot as plt

models = ["VGG-Face", "Facenet", "Facenet512",
          "OpenFace", "DeepFace", "DeepID", "ArcFace",
          "Dlib", "SFace"]

#face verification
result = DeepFace.verify(img1_path = "face-db/Elon/Elon-1.jpeg",
                        img2_path ="face-db/Elon/Elon-2.jpeg",
                         model_name=models[7])
print(result)
fig, axs = plt.subplots(1,2, figsize=(19, 5))
axs[0].imshow(plt.imread("face-db/Elon/Elon-1.jpeg"))
axs[1].imshow(plt.imread("face-db/Elon/Elon-2.jpeg"))
fig.suptitle(f"Verified: {result['verified']} - Distance: {result['distance']:.4f}")
plt.show()
