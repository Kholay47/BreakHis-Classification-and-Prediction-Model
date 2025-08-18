# 🧬 Breast Cancer Histopathology Image Classifier  

A deep learning project using **EfficientNet + PyTorch + Flask** to classify breast cancer histopathology images into 8 categories.   

---

## 📌 Features  

- ✅ Fine-tuned **EfficientNet model**  
- ✅ Trained on the **BreakHis dataset**  
- ✅ Flask-based **web interface** for predictions  
- ✅ Upload an image and get:  
  - Predicted class (tumor type)  
  - Confidence score (%)  
- ✅ Easy to deploy on **local machine or cloud**  

---

## 🏷️ Classes  

The model predicts one of the following 8 classes:  

- Adenosis  
- Ductal Carcinoma  
- Fibroadenoma  
- Lobular Carcinoma  
- Mucinous Carcinoma  
- Papillary Carcinoma  
- Phyllodes Tumor  
- Tubular Adenoma  

---

## ⚙️ Tech Stack  

- **PyTorch** – Deep learning model training  
- **Torchvision** – Image preprocessing  
- **Flask** – Web application framework  
- **PIL (Pillow)** – Image handling  
- **HTML / CSS (Jinja2 templates)** – User interface  

---

## 🖼️ Usage

- Open the web app in your browser

- Upload a histopathology image

- Click Predict

- View the predicted class + confidence score

- The uploaded image will also be displayed on the result page

## 📊 Model Training

- The model was trained using **EfficientNet-B0** (can be replaced with B1/B2).

- **Input image size**: 224x224

- **Optimizer**: Adam

- **Loss function**: CrossEntropyLoss

- **Achieved validation accuracy**: ~88%.
