# Single_Faces_Analyzation.py
# Analyze facial expression and decides the most probable outcome
import pandas as pd
from deepface import DeepFace
import matplotlib.pyplot as plt

# Analyze the image for emotions
result = DeepFace.analyze(img_path="face-db/Elon/Elon-1.jpeg", actions=['emotion'])

# Create a DataFrame from the emotion analysis results
emotion_df = pd.DataFrame(result[0]['emotion'], index=[0]).T.plot(kind="bar")

# Plot the emotion distribution as a bar plot
emotion_df.plot(kind="bar")
plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Probability")
plt.show()

# Print the result
print(result)
