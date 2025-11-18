# ♻️ Waste Classification Using CNN (MobileNetV2 + Transfer Learning)

This project is an AI-powered waste classification system built using **MobileNetV2**, **Transfer Learning**, and **Streamlit**.  
It classifies waste images into **6 categories** with high accuracy:

- Cardboard  
- Glass  
- Metal  
- Paper  
- Plastic  
- Trash  

The model achieves **92%+ test accuracy** and is deployed using a simple and interactive Streamlit web app.

---

## 🚀 Features

- Deep learning model based on MobileNetV2  
- Fine-tuned top 30 layers for high accuracy  
- Real-time image prediction using Streamlit  
- Easy-to-use interface  
- Fully reproducible code and trained `.h5` model included  

---

## 🧠 Model Overview

- **Architecture:** MobileNetV2 (pretrained on ImageNet)  
- **Input Size:** 224×224  
- **Optimizer:** Adam  
- **Loss:** Categorical Crossentropy  
- **Techniques Used:**  
  - Data Augmentation  
  - Transfer Learning  
  - Fine-tuning  
  - Dropout Regularization  

---

## 📊 Results

| Metric | Score |
|--------|--------|
| Training Accuracy | 90%+ |
| Validation Accuracy | 90%+ |
| Test Accuracy | **92%+** |

A confusion matrix and classification report were also generated to analyze model performance across all categories.

---

## 🧪 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/chetan-bisht-001/Waste_classification_using_CNN.git
cd Waste_classification_using_CNN
