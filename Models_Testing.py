# Faces_Verification.py
# Compares the stated images and returns similarity
from deepface import DeepFace
import matplotlib.pyplot as plt

models = ["VGG-Face", "Facenet", "Facenet512",
          "OpenFace", "DeepFace", "DeepID", "ArcFace",
          "Dlib", "SFace"]

# Dictionary to store results
results = {}

for model in models:
    #Selects which faces to verify
    result = DeepFace.verify(
        img1_path="face-db/Maxwell/Maxwell-1.jpeg",
        img2_path="face-db/Maxwell/Maxwell-2.jpeg",
        model_name=model,
    )
    results[model] = result

# Plot results
fig, axs = plt.subplots(len(models), 2, figsize=(15, 5 * len(models)))
fig.subplots_adjust(hspace=0.5)

for i, model in enumerate(models):
    result = results[model]
    # selects which faces to display
    axs[i, 0].imshow(plt.imread("face-db/Maxwell/Maxwell-1.jpeg"))
    axs[i, 1].imshow(plt.imread("face-db/Maxwell/Maxwell-2.jpeg"))
    axs[i, 0].axis("off")
    axs[i, 1].axis("off")
    axs[i, 0].set_title(f"Model: {model}", fontsize=10)
    axs[i, 1].set_title(f"Verified: {result['verified']} - Distance: {result['distance']:.4f}", fontsize=10)

plt.show()
