import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ==== Configuration ====
MODEL_PATH = "model3.pth"
NUM_CLASSES = 8
CLASS_NAMES = [
    "adenosis",
    "ductal_carcinoma",
    "fibroadenoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
    "phyllodes_tumor",
    "tubular_adenoma"
]

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== Model Loading ====
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ==== Image Transform ====
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== Prediction Function ====
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)

    predicted_class = CLASS_NAMES[pred.item()]
    return predicted_class

# ==== Main ====
if __name__ == "__main__":
    test_path = input("Enter image path or folder path: ").strip()

    if os.path.isdir(test_path):
        for img_file in os.listdir(test_path):
            if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(test_path, img_file)
                result = predict_image(img_path)
                print(f"{img_file} → {result}")
    else:
        result = predict_image(test_path)
        print(f"Prediction → {result}")
