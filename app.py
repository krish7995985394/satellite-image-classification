from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image

from train_eurosat import build_model

app = Flask(__name__)

MODEL_PATH = "eurosat_resnet18.pth"

# -----------------------------
# Load Model
# -----------------------------
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# Handle both checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    state = checkpoint["model_state_dict"]

    class_to_idx = checkpoint["class_to_idx"]

else:

    state = checkpoint

    class_to_idx = {
        'AnnualCrop': 0,
        'Forest': 1,
        'HerbaceousVegetation': 2,
        'Highway': 3,
        'Industrial': 4,
        'Pasture': 5,
        'PermanentCrop': 6,
        'Residential': 7,
        'River': 8,
        'SeaLake': 9
    }

# Convert index to class
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Build model
model = build_model(len(class_to_idx))

# Load trained weights
model.load_state_dict(state)

# Evaluation mode
model.eval()

# -----------------------------
# Image Preprocessing
# -----------------------------
transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():

    return render_template("index.html")

# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        })

    file = request.files["image"]

    # Open image
    image = Image.open(file).convert("RGB")

    # Preprocess image
    x = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():

        outputs = model(x)

        probs = torch.softmax(outputs, dim=1)[0]

    predictions = []

    # ALL predictions
    for i in range(len(probs)):

        predictions.append({

            "label": idx_to_class[i],

            "score": round(
                float(probs[i].item()) * 100,
                2
            )
        })

    # Sort by confidence
    predictions = sorted(
        predictions,
        key=lambda x: x["score"],
        reverse=True
    )

    return jsonify({
        file.filename: predictions
    })

# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=7860,
        debug=True
    )