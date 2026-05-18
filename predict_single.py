import torch
from torchvision import transforms
from PIL import Image
import argparse
import os
import math
import json
from train_eurosat import build_model

# ---- Config ----
MODEL_PATH = "eurosat_resnet18.pth"
CONF_THRESHOLD = 0.7
ENTROPY_THRESHOLD = 1.5


# ---- Load model ----
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    state = checkpoint["model_state_dict"]
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(len(class_to_idx))
    model.load_state_dict(state)
    model.eval()

    return model, idx_to_class


# ---- Preprocessing ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ---- Entropy ----
def entropy(probs):
    return -sum([p * math.log(p + 1e-10) for p in probs])


# ---- Core Prediction Logic ----
def predict_image(image_path, topk=5):
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    model, idx_to_class = load_model()

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    # Convert to Python list
    probs_list = probs.tolist()

    # Get top-k
    top_probs, top_idxs = probs.topk(topk)

    max_conf = top_probs[0].item()
    ent = entropy(probs_list)

    # 🔴 ---- Rejection Logic ----
    if max_conf < CONF_THRESHOLD or ent > ENTROPY_THRESHOLD:
        return {
            "status": "rejected",
            "reason": "Not a satellite image",
            "confidence": round(max_conf, 4),
            "entropy": round(ent, 4)
        }

    # ✅ Valid prediction
    predictions = []
    for prob, idx in zip(top_probs, top_idxs):
        predictions.append({
            "label": idx_to_class[idx.item()],
            "score": round(prob.item(), 4)
        })

    return {
        "status": "success",
        "confidence": round(max_conf, 4),
        "entropy": round(ent, 4),
        "predictions": predictions
    }


# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to image")
    args = parser.parse_args()

    result = predict_image(args.image_path)

    print(json.dumps(result, indent=2))