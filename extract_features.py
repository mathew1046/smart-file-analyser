import os
import cv2
import pytesseract
import pandas as pd
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def extract_image_features(folder_path, label):
    data = []

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        file_path = os.path.join(folder_path, file_name)
        try:
            img = Image.open(file_path)
            width, height = img.size
            file_size_kb = os.path.getsize(file_path) / 1024

            text = pytesseract.image_to_string(img)
            text_length = len(text.strip())

            cv_img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(cv_img, scaleFactor=1.1, minNeighbors=5)
            face_count = len(faces)

            data.append({
                "filename": file_name,
                "width": width,
                "height": height,
                "text_length": text_length,
                "face_count": face_count,
                "label": label
            })

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    imp_df = extract_image_features("data/important", 1)
    nonimp_df = extract_image_features("data/non-important", 0)
    full_df = pd.concat([imp_df, nonimp_df])
    full_df.to_csv("image_features.csv", index=False)
    print("✅ Feature extraction complete.")
