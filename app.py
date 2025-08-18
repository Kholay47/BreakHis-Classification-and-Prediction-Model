import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template
from PIL import Image
import os

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'adenosis', 'ductal_carcinoma', 'fibroadenoma',
    'lobular_carcinoma', 'mucinous_carcinoma',
    'papillary_carcinoma', 'phyllodes_tumor', 'tubular_adenoma'
]

def load_model(weights_path="model3.pth"):
    model = models.efficientnet_b0(weights=None)   # change if trained with b1/b2
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)
        predicted_class = CLASS_NAMES[preds.item()]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds.item()].item()

    return predicted_class, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400

        upload_folder = os.path.join("static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        pred_class, confidence = predict_image(file_path)

        return render_template("index.html",
                               prediction=pred_class,
                               confidence=confidence,
                               uploaded_image=file.filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
