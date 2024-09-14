# Faces_Recognition.py
# Compares the stated images and returns similarity
from deepface import DeepFace
import matplotlib.pyplot as plt

result = DeepFace.find(img_path="face-db/Elon/Elon-1.jpeg", db_path="face-db/")
print(result)

