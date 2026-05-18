# predict_batch.py
# Batch inference: walks an input folder (recursively), runs the trained model,
# and writes a CSV with top-k predictions and probabilities.
#
# Usage example:
#   python predict_batch.py --input_dir images --output_csv eurosat_predictions.csv --topk 3

import os
import csv
import torch
from torchvision import transforms
from PIL import Image
import argparse
from train_eurosat import build_model

MODEL_PATH = "eurosat_resnet18.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    class_to_idx = ckpt.get("class_to_idx", None) if isinstance(ckpt, dict) else None
    if class_to_idx is None:
        raise RuntimeError("class_to_idx missing in checkpoint")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    model = build_model(num_classes)
    model.load_state_dict(state)
    model.eval()
    return model, idx_to_class

def iter_images(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            lower = fn.lower()
            if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                yield os.path.join(dirpath, fn)

def predict_image(model, img_path, topk=3):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # 1 x C x H x W
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        top_probs, top_idxs = probs.topk(topk)
        return [(idx.item(), prob.item()) for prob, idx in zip(top_probs, top_idxs)]

def main(input_dir, output_csv, topk):
    model, idx_to_class = load_model()
    rows = []
    count = 0
    for img_path in iter_images(input_dir):
        try:
            preds = predict_image(model, img_path, topk=topk)
            # map idx->class
            mapped = [(idx_to_class[idx], float(prob)) for idx, prob in preds]
            # prepare CSV row: filepath, top1_label, top1_prob, top2_label, top2_prob, ...
            row = [img_path]
            for label, prob in mapped:
                row += [label, f"{prob:.6f}"]
            rows.append(row)
            count += 1
        except Exception as e:
            print(f"[warn] failed {img_path}: {e}")

    # Build header
    header = ["image_path"]
    for i in range(1, topk + 1):
        header += [f"top{i}_label", f"top{i}_prob"]

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)

    print(f"[done] Wrote {output_csv} ({count} images)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder with images (recursively searched)")
    parser.add_argument("--output_csv", required=True, help="Output CSV file path")
    parser.add_argument("--topk", type=int, default=3, help="Top-k predictions to write")
    args = parser.parse_args()
    main(args.input_dir, args.output_csv, args.topk)
