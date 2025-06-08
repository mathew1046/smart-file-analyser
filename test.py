import os
import cv2
import pytesseract
import pandas as pd
from PIL import Image
import joblib
import shutil
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

clf = joblib.load("important_classifier.pkl")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_features_from_image(img_path):
    try:
        img = Image.open(img_path)
        width, height = img.size
        text = pytesseract.image_to_string(img)
        text_length = len(text.strip())

        gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        face_count = len(faces)

        return [width, height, text_length, face_count]
    except:
        return None

input_folder = "images_to_sort"
important_folder = "sorted/important"
not_important_folder = "sorted/not_important"

os.makedirs(important_folder, exist_ok=True)
os.makedirs(not_important_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if not file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    file_path = os.path.join(input_folder, file_name)
    features = extract_features_from_image(file_path)
    print(f"Processing: {file_name}")
    print(f"Extracted Features: {features}")

    if features:
        prediction = clf.predict([features])[0]
        print(f"Prediction: {prediction}")
        dest_folder = important_folder if prediction == 1 else not_important_folder
        shutil.copy(file_path, os.path.join(dest_folder, file_name))
        print(f"{file_name} → {'Important' if prediction == 1 else 'Not Important'}")

print("✅ Sorting complete.")
