# plot_qnn_learning_curve.py
from pathlib import Path
import matplotlib.pyplot as plt


def plot_loss_vs_steps(loss_history, output_path):
    """
    Loss curve for a single QNN run.
    Theme: Blue line, Grey grid
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = list(range(1, len(loss_history) + 1))
    plt.figure(figsize=(8.8, 4.8))
    plt.plot(steps, loss_history, linewidth=2.5)
    plt.scatter(steps[::max(1, len(steps)//15)], [loss_history[i-1] for i in steps[::max(1, len(steps)//15)]],
                s=18, zorder=3)

    plt.title("QNN Training Curve: Loss vs Steps")
    plt.xlabel("Training step")
    plt.ylabel("Loss (binary cross-entropy)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_path}")