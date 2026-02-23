# 🌿 AI Plant Disease Detection System

A Deep Learning-based web application that detects plant leaf diseases using Transfer Learning (InceptionV3).

---

## 🚀 Features

- Detects:
  - ✅ Healthy Leaves
  - 🍂 Rust Infection
  - 🌫 Powdery Mildew
- Confidence score display
- Full probability breakdown
- Low-confidence detection handling
- Interactive Flask-based UI

---

## 🧠 Model Details

- Architecture: InceptionV3 (Transfer Learning)
- Input Size: 299x299
- Training Images: 1322
- Validation Images: 60
- Test Images: 150
- Final Test Accuracy: **96.67%**

The model was trained using data augmentation and fine-tuning for improved generalization.

---

## 📊 Model Performance

Test Accuracy: **96.67%**

Confusion Matrix Summary:

| Class     | Precision | Recall |
|-----------|-----------|--------|
| Healthy   | 0.93      | 1.00   |
| Powdery   | 0.98      | 0.92   |
| Rust      | 1.00      | 0.98   |

The model shows strong generalization on unseen test data.

---

## 🖥️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Flask
- HTML / CSS
- Scikit-learn
- Matplotlib

---

## 📂 Project Structure
plant-disease-detection/
│
├── app.py
├── train_model.py
├── test_model.py
├── best_model.keras
├── requirements.txt
│
├── templates/
│ └── index.html
│
├── static/
│
└── .gitignore



---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
cd plant-disease-detection