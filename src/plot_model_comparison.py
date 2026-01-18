# src/plot_model_comparison.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {missing}")


def plot_model_comparison(csv_path: Path, out_path: Path) -> None:
    """
    Creates ONE output image with TWO stacked graphs:

      Graph 1 (top): Accuracy / Precision / Recall / F1 in ONE grouped-bar chart
      Graph 2 (bottom): Inference Time (sec_per_pred)

    Models included:
      Logistic Regression, SVM, Neural Network, QNN (5 qubits)

    Expected CSV columns:
      model, accuracy, precision, recall, f1, sec_per_pred
    """
    df = pd.read_csv(csv_path)
    df["model"] = df["model"].astype(str).str.strip()

    _require_cols(df, ["model", "accuracy", "precision", "recall", "f1", "sec_per_pred"])

    for c in ["accuracy", "precision", "recall", "f1", "sec_per_pred"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep_models = [
        "Logistic Regression",
        "SVM",
        "Neural Network",
        "QNN (5 qubits)",
    ]

    df = df[df["model"].isin(keep_models)].copy()

    missing_models = [m for m in keep_models if m not in set(df["model"])]
    if missing_models:
        raise ValueError(
            f"These models are missing in CSV: {missing_models}\n"
            f"Found models: {sorted(df['model'].unique().tolist())}"
        )

    # enforce order
    df = df.set_index("model").loc[keep_models].reset_index()

    models = df["model"].tolist()
    x = np.arange(len(models))

    acc = df["accuracy"].to_numpy()
    prec = df["precision"].to_numpy()
    rec = df["recall"].to_numpy()
    f1 = df["f1"].to_numpy()
    lat = df["sec_per_pred"].to_numpy()

    # ---- Figure: two stacked graphs in one image ----
    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.35)

    # Graph 1: Accuracy + Precision + Recall + F1 (grouped bars)
    ax1 = fig.add_subplot(gs[0, 0])
    width = 0.20

    ax1.bar(x - 1.5 * width, acc,  width, label="Accuracy")
    ax1.bar(x - 0.5 * width, prec, width, label="Precision")
    ax1.bar(x + 0.5 * width, rec,  width, label="Recall")
    ax1.bar(x + 1.5 * width, f1,   width, label="F1-score")

    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Score")
    ax1.set_title("Accuracy / Precision / Recall / F1 Comparison (Selected Models)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.25, axis="y")

    # Value labels (comment out if you want a cleaner figure)
    def _annotate(ax, values, offset):
        for i, v in enumerate(values):
            if np.isfinite(v):
                ax.text(i + offset, v + 0.02, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9)

    _annotate(ax1, acc,  -1.5 * width)
    _annotate(ax1, prec, -0.5 * width)
    _annotate(ax1, rec,  +0.5 * width)
    _annotate(ax1, f1,   +1.5 * width)

    # Graph 2: Inference Time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(x, lat)

    ax2.set_title("Inference Time (seconds per prediction) — Selected Models")
    ax2.set_ylabel("Seconds")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=0)
    ax2.grid(alpha=0.25, axis="y")

    max_lat = np.nanmax(lat)
    pad = 0.02 * max_lat if max_lat > 0 else 0.0

    for i, v in enumerate(lat):
        if np.isfinite(v):
            ax2.text(i, v + pad, f"{v:.2e}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Model Comparison for Advance-Fee Fraud Detection", fontsize=16)
    fig.subplots_adjust(top=0.90)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined figure: {out_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    csv_in = project_root / "results" / "results_all.csv"
    out_png = project_root / "results" / "model_comparison.png"

    plot_model_comparison(csv_in, out_png)