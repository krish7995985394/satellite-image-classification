# train_eurosat.py

import os
import time
import argparse
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models, datasets

from training_logger import log_epoch

ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(ROOT, "data_eurosat", "eurosat", "2750")
MODEL_PATH = os.path.join(ROOT, "eurosat_resnet18.pth")


def get_dataloaders(batch_size=64, subset_size=10000, val_ratio=0.2, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transform)

    labels = [lbl for (_, lbl) in full_dataset.samples]
    num_total = len(labels)
    print(f"Total images available in dataset: {num_total}")

    if subset_size < num_total:
        class_to_indices = defaultdict(list)
        for idx, lbl in enumerate(labels):
            class_to_indices[lbl].append(idx)

        num_classes = len(class_to_indices)
        base = subset_size // num_classes
        rnd = random.Random(42)

        sampled_indices = []
        for cls_idx, idx_list in class_to_indices.items():
            sampled_indices.extend(rnd.sample(idx_list, base))

        full_dataset = Subset(full_dataset, sampled_indices)
        print(f"Using subset of size: {len(sampled_indices)}")

    from sklearn.model_selection import train_test_split
    indices = list(range(len(full_dataset)))

    # ✅ FIX: handle val_ratio = 0 (for --full)
    if val_ratio == 0.0:
        train_idx = indices
        val_idx = indices
    else:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=42
        )

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    class_to_idx = full_dataset.dataset.class_to_idx if isinstance(full_dataset, Subset) else full_dataset.class_to_idx

    return train_loader, val_loader, class_to_idx


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(num_epochs=30, lr=0.0001, subset_size=10000, batch_size=64, num_workers=4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, class_to_idx = get_dataloaders(
        batch_size, subset_size, 0.2, num_workers
    )

    num_classes = len(class_to_idx)
    print("Classes:", list(class_to_idx.keys()))

    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch [{epoch}/{num_epochs}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("✅ Best model saved!")

    print("Training Complete")
    print("Best Accuracy:", best_val_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    train(
        num_epochs=args.num_epochs,
        lr=args.lr,
        subset_size=args.subset_size,
        batch_size=args.batch_size
    )