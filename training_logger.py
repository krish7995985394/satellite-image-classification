# training_logger.py
# Helper for saving epoch metrics into a CSV file.

import csv
import os

def log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, csv_path="training_log.csv"):
    header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]

    file_exists = os.path.isfile(csv_path)
    write_header = not file_exists or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        writer.writerow([
            epoch,
            float(train_loss),
            float(train_acc),
            float(val_loss),
            float(val_acc)
        ])
