# make_results_png.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def make_results_png(results_df, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    models = results_df["model"].tolist()

    metrics = ["accuracy", "precision", "recall", "f1"]
    titles  = ["Accuracy", "Precision", "Recall", "F1"]
    ylabels = ["accuracy", "precision", "recall", "f1"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # --- Top: Accuracy ---
    ax = axes[0]
    vals = results_df["accuracy"].values
    ax.bar(models, vals)
    ax.set_title("Accuracy Comparison Across Models")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")

    # --- Middle: Precision/Recall/F1 together (3 bars groups) ---
    ax = axes[1]
    x = np.arange(len(models))
    width = 0.25

    prec = results_df["precision"].values
    rec  = results_df["recall"].values
    f1   = results_df["f1"].values

    ax.bar(x - width, prec, width, label="Precision")
    ax.bar(x,         rec,  width, label="Recall")
    ax.bar(x + width, f1,   width, label="F1")

    ax.set_title("Precision / Recall / F1 Comparison")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10)
    ax.legend()

    # label bars
    for i in range(len(models)):
        ax.text(x[i] - width, prec[i] + 0.02, f"{prec[i]:.3f}", ha="center")
        ax.text(x[i],         rec[i]  + 0.02, f"{rec[i]:.3f}",  ha="center")
        ax.text(x[i] + width, f1[i]   + 0.02, f"{f1[i]:.3f}",   ha="center")

    # --- Bottom: Inference time ---
    ax = axes[2]
    tvals = results_df["sec_per_pred"].values
    ax.bar(models, tvals)
    ax.set_title("Inference Time Comparison (sec per prediction)")
    ax.set_ylabel("sec_per_pred")
    for i, v in enumerate(tvals):
        ax.text(i, v + (max(tvals) * 0.02 if max(tvals) > 0 else 0.0001), f"{v:.3e}", ha="center")

    fig.suptitle("Model Comparison (6 Features)", fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)