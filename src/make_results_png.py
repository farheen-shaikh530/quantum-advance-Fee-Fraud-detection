# src/make_results_png.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .plot_style import apply, THEME  # <-- important: relative import


def make_results_png(results_df, out_path):
    """
    Creates a publication-ready PNG comparing models:
    - Accuracy
    - Precision / Recall / F1
    - Inference latency
    Theme: Blue / Yellow / Grey
    """
    apply()  # apply theme once at the start

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models = results_df["model"].astype(str).tolist()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # -----------------------
    # (1) Accuracy
    # -----------------------
    ax = axes[0]
    acc = results_df["accuracy"].values
    ax.bar(models, acc, color=THEME["BLUE"])
    ax.set_title("Accuracy Comparison Across Models")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(acc):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", color=THEME["GREY"])

    # -----------------------
    # (2) Precision / Recall / F1
    # -----------------------
    ax = axes[1]
    x = np.arange(len(models))
    width = 0.25

    prec = results_df["precision"].values
    rec  = results_df["recall"].values
    f1   = results_df["f1"].values

    ax.bar(x - width, prec, width, label="Precision", color=THEME["GREY"])
    ax.bar(x,         rec,  width, label="Recall",    color=THEME["BLUE"])
    ax.bar(x + width, f1,   width, label="F1-score",  color=THEME["YELLOW"])

    ax.set_title("Precision / Recall / F1 Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10, ha="right")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.3)

    # -----------------------
    # (3) Inference latency
    # -----------------------
    ax = axes[2]
    latency = results_df["sec_per_pred"].values
    ax.bar(models, latency, color=THEME["GREY"])
    ax.set_title("Inference Time (seconds per prediction)")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.3)

    top = float(np.max(latency)) if len(latency) else 0.0
    for i, v in enumerate(latency):
        ax.text(i, v + (top * 0.02 if top > 0 else 1e-6), f"{v:.2e}", ha="center")

    fig.suptitle("Model Comparison for Advance-Fee Fraud Detection", fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved results figure to: {out_path}")