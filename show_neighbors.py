# show_neighbors.py
# Load embeddings.npy + filenames.csv and print top-k nearest neighbors (cosine similarity)

import numpy as np
import csv
import os
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(__file__)
EMB = os.path.join(ROOT, "embeddings.npy")
CSV = os.path.join(ROOT, "filenames.csv")

def load_filenames(csvpath):
    with open(csvpath, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        rows = [r[0] for r in reader]
    return rows

def main(k=3):
    if not os.path.exists(EMB) or not os.path.exists(CSV):
        print("Missing embeddings.npy or filenames.csv — run extract_embeddings.py first.")
        return
    embeddings = np.load(EMB)
    files = load_filenames(CSV)
    sims = cosine_similarity(embeddings)
    n = embeddings.shape[0]
    for i in range(n):
        scores = sims[i]
        idx = (-scores).argsort()  # descending
        nn = [j for j in idx if j != i][:k]
        print(f"\nFile: {files[i]}")
        for rank, j in enumerate(nn, start=1):
            print(f"  {rank}. {files[j]}  (sim={scores[j]:.4f})")

if __name__ == "__main__":
    main(k=3)
