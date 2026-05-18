# diagnose_eval_fix.py
# Diagnostic with num_workers=0 to avoid multiprocessing spawn issues on Windows.

import os
import torch
from collections import Counter
from train_eurosat import get_dataloaders, build_model, MODEL_PATH

def run_diagnostic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subset_size = 10000
    val_ratio = 0.2
    batch_size = 64
    num_workers = 0   # IMPORTANT: avoid multiprocessing spawn on Windows

    # get loaders (this uses the same logic as training)
    train_loader, val_loader, class_to_idx, train_size, val_size = get_dataloaders(
        batch_size=batch_size, subset_size=subset_size, val_ratio=val_ratio, num_workers=num_workers
    )

    print("class_to_idx:", class_to_idx)
    classes = list(class_to_idx.keys()) if class_to_idx else None
    print("classes list (len):", (len(classes) if classes else None))

    # count true labels in validation set
    true_counts = Counter()
    all_true = []
    for imgs, labels in val_loader:
        for l in labels.tolist():
            true_counts[int(l)] += 1
            all_true.append(int(l))

    print("Validation label counts (index:count):")
    for k in sorted(true_counts.keys()):
        print(f"  {k}: {true_counts[k]}")

    # load model
    num_classes = len(classes) if classes else 1
    model = build_model(num_classes)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # robust load
    try:
        model.load_state_dict(state)
        print("Model: strict load OK")
    except Exception as e:
        print("Model: strict load failed:", e)
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_sd and state[k].shape == model_sd[k].shape}
        model_sd.update(filtered)
        model.load_state_dict(model_sd)
        print("Model: partial load applied")

    model.to(device)
    model.eval()

    # predictions
    pred_counts = Counter()
    pairs = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().tolist()
            for t, p in zip(labels.tolist(), preds):
                pred_counts[int(p)] += 1
                if len(pairs) < 40:
                    pairs.append((int(t), int(p)))

    print("Predicted label counts (index:count):")
    for k in sorted(pred_counts.keys()):
        print(f"  {k}: {pred_counts[k]}")

    print("\nFirst 40 (true_index, pred_index) examples:")
    for t, p in pairs:
        print(f"  {t} -> {p}")

    if classes:
        print("\nIndex -> class mapping:")
        for i in range(len(classes)):
            print(f"  {i}: {classes[i]}")

if __name__ == "__main__":
    run_diagnostic()
