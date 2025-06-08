import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("image_features.csv")

X = df[['width', 'height', 'text_length', 'face_count']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")

import joblib
joblib.dump(clf, "important_classifier.pkl")
print("✅ Model saved as important_classifier.pkl")
