# plot_qnn_analysis.py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_qnn_all_metrics_vs_qubits(
    qubits,
    accuracy,
    precision,
    recall,
    f1,
    output_path="results/qnn_all_metrics_vs_qubits.png"
):
    out = Path(output_path)
    out.parent.mkdir(exist_ok=True)

    plt.figure(figsize=(7.5, 5.2))
    plt.plot(qubits, accuracy, marker="o", linewidth=2, label="Accuracy")
    plt.plot(qubits, precision, marker="o", linewidth=2, label="Precision")
    plt.plot(qubits, recall, marker="o", linewidth=2, label="Recall")
    plt.plot(qubits, f1, marker="o", linewidth=2, label="F1-score")

    plt.xlabel("Number of Qubits")
    plt.ylabel("Metric Value")
    plt.title("QNN Performance vs Number of Qubits (4–6)")
    plt.xticks(qubits)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Saved: {out}")


def plot_qnn_vs_mlp(
    mlp_metrics,
    qnn_metrics,
    output_path="results/qnn_vs_mlp.png"
):
    """
    Bar chart: compare MLP vs QNN across Accuracy/Precision/Recall/F1
    mlp_metrics = [acc, prec, rec, f1]
    qnn_metrics = [acc, prec, rec, f1]
    """
    out = Path(output_path)
    out.parent.mkdir(exist_ok=True)

    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, mlp_metrics, width, label="Neural Network (MLP)")
    plt.bar(x + width/2, qnn_metrics, width, label="QNN (6 qubits)")

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Metric Value")
    plt.title("QNN vs Classical Neural Network (MLP)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Saved: {out}")