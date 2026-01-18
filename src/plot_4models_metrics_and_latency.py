# src/plot_4models_metrics_and_latency.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


KEEP_MODELS = [
    "Logistic Regression",
    "SVM",
    "Neural Network",
    "QNN (5 qubits)",
]


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {missing}")


def _load_and_filter(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["model"] = df["model"].astype(str).str.strip()

    _require_cols(df, ["model", "accuracy", "precision", "recall", "f1", "sec_per_pred"])

    for c in ["accuracy", "precision", "recall", "f1", "sec_per_pred"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["model"].isin(KEEP_MODELS)].copy()

    missing_models = [m for m in KEEP_MODELS if m not in set(df["model"])]
    if missing_models:
        raise ValueError(
            f"These models are missing in CSV: {missing_models}\n"
            f"Found models: {sorted(df['model'].unique().tolist())}"
        )

    df = df.set_index("model").loc[KEEP_MODELS].reset_index()
    return df


def plot_metrics_one_graph(df: pd.DataFrame, out_path: Path) -> None:
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.20

    acc = df["accuracy"].to_numpy()
    prec = df["precision"].to_numpy()
    rec = df["recall"].to_numpy()
    f1 = df["f1"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - 1.5 * width, acc,  width, label="Accuracy")
    ax.bar(x - 0.5 * width, prec, width, label="Precision")
    ax.bar(x + 0.5 * width, rec,  width, label="Recall")
    ax.bar(x + 1.5 * width, f1,   width, label="F1-score")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy / Precision / Recall / F1 Comparison (Selected Models)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    # annotate values
    def annotate(vals, offset):
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i + offset, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    annotate(acc, -1.5 * width)
    annotate(prec, -0.5 * width)
    annotate(rec,  +0.5 * width)
    annotate(f1,   +1.5 * width)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_inference_time(df: pd.DataFrame, out_path: Path) -> None:
    models = df["model"].tolist()
    x = np.arange(len(models))
    lat = df["sec_per_pred"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, lat)

    ax.set_title("Inference Time (seconds per prediction) — Selected Models")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.grid(axis="y", alpha=0.25)

    max_lat = np.nanmax(lat)
    pad = 0.02 * max_lat if max_lat > 0 else 0.0

    for i, v in enumerate(lat):
        if np.isfinite(v):
            ax.text(i, v + pad, f"{v:.2e}", ha="center", va="bottom", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_in = project_root / "results" / "results_all.csv"

    df = _load_and_filter(csv_in)

    plot_metrics_one_graph(df, project_root / "results" / "metrics_4models.png")
    plot_inference_time(df, project_root / "results" / "inference_time_4models.png")


if __name__ == "__main__":
    main()