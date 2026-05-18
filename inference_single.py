# inference_single.py
import os
import sys
import torch
from torchvision import transforms, models
from PIL import Image
import argparse
import json

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "eurosat_resnet18.pth")
DATA_ROOT = os.path.join(ROOT, "data_eurosat")

def load_class_mapping():
    # The saved checkpoint contains class_to_idx mapping; invert to idx->class
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    class_to_idx = ckpt.get("class_to_idx")
    if class_to_idx is None:
        raise RuntimeError("class_to_idx not found in checkpoint")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class, ckpt

def build_model(num_classes, ckpt):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)

def predict(img_path, topk=3, device="cpu"):
    idx_to_class, ckpt = load_class_mapping()
    model = build_model(len(idx_to_class), ckpt)
    model.to(device)
    x = preprocess_image(img_path).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)
        topk_probs, topk_idx = probs.topk(topk, dim=1)
    results = []
    for p, i in zip(topk_probs[0].tolist(), topk_idx[0].tolist()):
        results.append({"class": idx_to_class[i], "prob": float(p)})
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    args = parser.parse_args()

    res = predict(args.image, topk=args.topk, device=args.device)
    print("Predictions:")
    for r in res:
        print(f"  {r['class']:20s} {r['prob']:.4f}")
