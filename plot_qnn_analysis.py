# plot_qnn_analysis.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt



def plot_qnn_all_metrics_vs_qubits(
    qubits,
    accuracy,
    precision,
    recall,
    f1,
    output_path="results/qnn_all_metrics_vs_qubits.png"
):
    """
    Line plot: Accuracy/Precision/Recall/F1 vs number of qubits
    Theme: Blue/Yellow/Grey
    """

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.8, 5.2))

    # Blue / Yellow / Grey usage
    plt.plot(qubits, accuracy, marker="o", linewidth=2.5)
    plt.plot(qubits, f1,       marker="o", linewidth=2.5)
    plt.plot(qubits, precision, marker="o", linewidth=2.2)
    plt.plot(qubits, recall,    marker="o", linewidth=2.2) # neutral black

    plt.xlabel("Number of Qubits")
    plt.ylabel("Metric Value")
    plt.title("QNN Performance vs Number of Qubits (4–6)")
    plt.xticks(qubits)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend(frameon=True)

    # annotate points
    for x, y in zip(qubits, accuracy):
        plt.text(x, y + 0.02, f"{y:.2f}", ha="center")
    for x, y in zip(qubits, f1):
        plt.text(x, y - 0.05, f"{y:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_qnn_vs_mlp(
    mlp_metrics,
    qnn_metrics,
    output_path="results/qnn_vs_mlp.png"
):
    """
    Bar chart: compare MLP vs QNN across Accuracy/Precision/Recall/F1
    Theme: Blue (MLP) vs Yellow (QNN), grey styling
    """

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(8.4, 5.2))
    b1 = plt.bar(x - width/2, mlp_metrics, width, label="Neural Network (MLP)")
    b2 = plt.bar(x + width/2, qnn_metrics, width, label="QNN (Selected Qubits)")

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Metric Value")
    plt.title("QNN vs Classical Neural Network (MLP)")
    plt.grid(axis="y")
    plt.legend(frameon=True)

    # value labels
   
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")