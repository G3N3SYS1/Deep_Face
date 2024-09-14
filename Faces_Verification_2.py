import os
import pandas as pd
from deepface import DeepFace
from glob import glob

# Use only the 'dlib' model
models = ["Dlib"]

# Path to the fixed image
fixed_image_path = "face-db/Elon/Elon-1.jpeg"

# Get all image files in the directory (recursive)
image_paths = glob("face-db/**/*", recursive=True)

# List to store results
results = []

# Debug print to check image paths
print(f"Fixed image path: {fixed_image_path}")
print(f"Found {len(image_paths)} images.")

# Iterate over each image in the directory
for image_path in image_paths:
    if image_path.endswith(('.jpeg', '.jpg', '.png')) and image_path != fixed_image_path:
        for model_name in models:
            try:
                print(f"Comparing {fixed_image_path} with {image_path} using {model_name}")
                result = DeepFace.verify(img1_path=fixed_image_path,
                                         img2_path=image_path,
                                         model_name=model_name)
                results.append({
                    'model': model_name,
                    'image': os.path.basename(image_path),
                    'verification': result['verified']
                })
                print(f"Result for {image_path}: {result['verified']}")
            except Exception as e:
                print(f"Error comparing {fixed_image_path} with {image_path} using {model_name}: {e}")

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Print the DataFrame
print("\nVerification Results:")
print(df)

# Optionally, save the results to a CSV file if needed
df.to_csv("verification_results.csv", index=False)
