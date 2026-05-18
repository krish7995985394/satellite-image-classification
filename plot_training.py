# plot_training.py

import argparse
import csv
import os
import matplotlib.pyplot as plt


def read_metrics(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))

    return epochs, train_loss, train_acc, val_loss, val_acc


def plot_curves(csv_path, out_png):
    epochs, train_loss, train_acc, val_loss, val_acc = read_metrics(csv_path)

    plt.figure(figsize=(12, 8))

    # 🔹 LOSS CURVE
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, marker="o", linewidth=2, label="Train Loss")
    plt.plot(epochs, val_loss, marker="o", linewidth=2, label="Validation Loss")
    plt.title("Learning Curve - Loss", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # 🔹 ACCURACY CURVE
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_acc, marker="o", linewidth=2, label="Train Accuracy")
    plt.plot(epochs, val_acc, marker="o", linewidth=2, label="Validation Accuracy")
    plt.title("Learning Curve - Accuracy", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()

    print(f"[plotting] saved: {out_png}")

    # 🔹 Summary
    try:
        best_val = max(val_acc)
        best_epoch = epochs[val_acc.index(best_val)]

        summary = (
            f"Best validation accuracy: {best_val:.4f} at epoch {best_epoch}\n"
            f"Final training accuracy: {train_acc[-1]:.4f}\n"
            f"Final validation accuracy: {val_acc[-1]:.4f}\n"
            f"Epochs: {len(epochs)}\n"
        )
    except Exception:
        summary = "No validation metrics available.\n"

    with open("training_log_summary.txt", "w") as f:
        f.write(summary)

    print("[plotting] summary written: training_log_summary.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="training_log.csv")
    parser.add_argument("--out", type=str, default="training_curves.png")
    args = parser.parse_args()

    plot_curves(args.csv, args.out)


if __name__ == "__main__":
    main()