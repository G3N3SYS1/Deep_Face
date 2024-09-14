# Multiple_Faces_Analyzation.py
# Returns results of emotions in bar charts

import os
import pandas as pd
from deepface import DeepFace
import matplotlib.pyplot as plt
from glob import glob

# Directory containing the images
image_dir = glob("face-db/**/*", recursive=True)
#image_dir = "face-db/*"

# List to store results
results = []

# Iterate over each image in the directory
for image_path in image_dir:
    if image_path.endswith(('.jpeg', '.jpg', '.png')):
        # Analyze the image for emotions
        try:
            result = DeepFace.analyze(img_path=image_path, actions=['emotion'])
            emotion_data = result[0]['emotion']
            emotion_data['image'] = os.path.basename(image_path)  # Add image name to the data
            results.append(emotion_data)
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Plot the emotion distribution for each image
fig, ax = plt.subplots(figsize=(12, 8))
df.set_index('image').plot(kind='bar', ax=ax)
plt.title("Emotion Distribution Across Images")
plt.xlabel("Image")
plt.ylabel("Probability")
plt.xticks(rotation=90)
plt.show()

# Print the results
print(df)
