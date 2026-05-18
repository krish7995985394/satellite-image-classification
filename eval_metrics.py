# eval_metrics.py

import os
import argparse
import torch
import csv
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix

from train_eurosat import get_dataloaders, build_model

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "eurosat_resnet18.pth")


def load_checkpoint(device):
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt.get("class_to_idx")
    return ckpt, None


def evaluate(loader, model, device):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            y_pred.extend(preds.cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", type=int, default=10000)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Subset size: {args.subset_size}")
    print(f"Val ratio: {args.val_ratio}")

    train_loader, val_loader, class_to_idx = get_dataloaders(
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers
    )

    classes = list(class_to_idx.keys())
    print("Classes:", classes)

    # ✅ Use ONLY validation set
    eval_loader = val_loader

    state_dict, _ = load_checkpoint(device)

    model = build_model(len(classes))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 🔍 Get predictions
    y_true, y_pred = evaluate(eval_loader, model, device)

    # 🚀 ✅ BALANCE TO EXACT 200 PER CLASS
    class_samples = defaultdict(list)

    for t, p in zip(y_true, y_pred):
        class_samples[t].append((t, p))

    new_y_true = []
    new_y_pred = []

    for cls in sorted(class_samples.keys()):
        samples = class_samples[cls]

        if len(samples) < 200:
            print(f"⚠️ Warning: {classes[cls]} has only {len(samples)} samples")

        selected = samples[:200]

        for t, p in selected:
            new_y_true.append(t)
            new_y_pred.append(p)

    y_true = new_y_true
    y_pred = new_y_pred

    # 📊 Accuracy
    total = len(y_true)
    correct = sum([t == p for t, p in zip(y_true, y_pred)])
    acc = correct / total

    print(f"\nBalanced Accuracy = {acc:.4f} ({correct}/{total}) -> {acc*100:.2f}%\n")

    # 📊 Classification Report
    report = classification_report(
        y_true, y_pred,
        target_names=classes,
        digits=4
    )

    print(report)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    print("Wrote classification_report.txt")

    # 📊 Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    with open("confusion_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + classes)
        for i, row in enumerate(cm):
            writer.writerow([classes[i]] + row.tolist())

    print("Wrote confusion_matrix.csv")
    print("Done.")


if __name__ == "__main__":
    main()