import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the confusion matrix generated for validation set
cm = pd.read_csv("confusion_matrix.csv", index_col=0)

plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.5,
    linecolor='gray'
)

plt.title("Confusion Matrix (Validation Set - Accuracy: 98.41%)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("confusion_matrix_validation.png", dpi=300)
plt.show()