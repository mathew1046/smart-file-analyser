# 📁 Smart File Sorter using Machine Learning

This is a Python-based desktop utility that classifies images as **Important** or **Not Important** using a trained machine learning model. It's designed to help users automatically sort their image folders by identifying:

- 📸 **Important** = images with people (e.g. family/friends)
- 🗒️ **Not Important** = screenshots, posters, or text-based images

## 🚀 Features

- Detects faces using OpenCV's Haar cascades
- Extracts text using Tesseract OCR
- Trained RandomForestClassifier with >90% accuracy
- Automatically copies sorted images into `sorted/important` and `sorted/not_important` folders
- Works on any folder of `.jpg`, `.png`, or `.jpeg` images

## 📂 Project Structure

```
smart-file-analyser/
├── extract_features.py       # Extracts features from labeled images
├── image_features.csv        # Generated dataset for training
├── train_model.py            # Trains ML model and saves it
├── important_classifier.pkl  # Saved ML model (Random Forest)
├── predict_new_images.py     # Predicts and sorts new images
├── images_to_sort/           # Place new images here
├── sorted/
│   ├── important/
│   └── not_important/
```

## 🧠 How it Works

For each image, the model extracts:

| Feature         | Description                           |
|----------------|---------------------------------------|
| `width, height` | Image dimensions                     |
| `text_length`   | Number of OCR-detected characters    |
| `face_count`    | Number of detected faces (if any)    |

These features are then classified using a Random Forest model trained on labeled image data.

## ⚙️ Requirements

Install dependencies:

```bash
pip install opencv-python pytesseract scikit-learn pandas joblib pillow
```

## 🔧 Setting Up Tesseract (Important!)

You must install [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) and set its path manually.

In your code, add this line at the top:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

> 📌 Modify the path according to where Tesseract is installed on your system.

## 🧪 How to Use

### Step 1: Label and Extract Features

Create two folders with sample images:

```
important/         # Images with people
not_important/     # Screenshots or posters
```

Run:
```bash
python extract_features.py
```

### Step 2: Train the Model

```bash
python train_model.py
```

> This creates `important_classifier.pkl`

### Step 3: Sort New Images

Place unsorted images in `images_to_sort/` and run:

```bash
python predict_new_images.py
```

Sorted results appear in the `sorted/` folder.

## 📊 Accuracy

Achieved **90% classification accuracy** on test data. Reliable for basic use cases like separating photos from memes/screenshots.

## 🧠 Notes

- Currently supports `.jpg`, `.png`, `.jpeg`
- You can expand it later with more advanced models (like CNN or MobileNetV2)

## 📘 Author

Built by [Mathew Joseph](https://github.com/mathew1046)  
🚀 June 2025 · Weekend project with ML + OpenCV + OCR
