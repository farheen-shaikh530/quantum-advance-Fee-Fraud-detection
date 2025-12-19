# plot_qnn_learning_curve.py
from pathlib import Path
import matplotlib.pyplot as plt

def plot_loss_vs_steps(loss_history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    steps = list(range(1, len(loss_history) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(steps, loss_history, marker="o", linewidth=2)
    plt.title("QNN Training Curve: Loss vs Steps")
    plt.xlabel("Training step")
    plt.ylabel("Loss (binary cross-entropy)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()