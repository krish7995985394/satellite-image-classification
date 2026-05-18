# extract_embeddings.py
# Minimal, deterministic pipeline: load images from images/, run through pretrained ResNet18,
# export embeddings (numpy) and filenames (csv). CPU-only friendly.

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import os
import csv
from tqdm import tqdm

ROOT = os.path.dirname(__file__)
IMG_DIR = os.path.join(ROOT, "images")
OUT_EMB = os.path.join(ROOT, "embeddings.npy")
OUT_CSV = os.path.join(ROOT, "filenames.csv")

# Image transform should match the ResNet training preprocessing
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def list_images(img_dir):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return [os.path.join(img_dir, f) for f in files]

def load_image(path):
    return Image.open(path).convert("RGB")

def main():
    device = torch.device("cpu")  # deterministic, works everywhere
    print("Device:", device)
    # Load pre-trained ResNet-18 and use it as a feature extractor (remove final fc)
    model = models.resnet18(pretrained=True)
    model.eval()
    # Remove the final classification layer: produce 512-d features
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device)

    img_paths = list_images(IMG_DIR)
    if not img_paths:
        print("No images found in", IMG_DIR)
        return

    embeddings = []
    filenames = []
    with torch.no_grad():
        for p in tqdm(img_paths, desc="Images"):
            try:
                img = load_image(p)
                inp = transform(img).unsqueeze(0).to(device)
                out = model(inp)                # shape: (1, 512, 1, 1)
                out = out.reshape(out.size(0), -1).cpu().numpy()  # (1,512)
                embeddings.append(out[0])
                filenames.append(os.path.basename(p))
            except Exception as e:
                print("ERROR loading", p, "->", e)

    embeddings = np.stack(embeddings, axis=0)
    np.save(OUT_EMB, embeddings)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename"])
        for fn in filenames:
            writer.writerow([fn])

    print("Wrote embeddings:", OUT_EMB, "shape:", embeddings.shape)
    print("Wrote filenames:", OUT_CSV, "count:", len(filenames))

if __name__ == "__main__":
    main()
