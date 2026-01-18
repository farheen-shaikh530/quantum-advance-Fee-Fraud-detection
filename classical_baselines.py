# classical_baselines.py
# ------------------------------------------------------------
# Trains classical baselines (LR, SVM-RBF, MLP) using:
# - same feature engineering as your run_all.py
# - stratified train/test split (80/20, random_state=42)
# - MinMax scaling
# Saves:
# - results/results_classical.csv
# - results/classical_metrics_bar.png
# - results/classical_confusion_matrices.png
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# -----------------------
# Paths
# -----------------------
ROOT = Path(__file__).resolve().parents[0]  # put this file at project root
DATA_FILE = ROOT / "transactions_2000.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_OUT = RESULTS_DIR / "results_classical.csv"
PNG_BAR = RESULTS_DIR / "classical_metrics_bar.png"
PNG_CM = RESULTS_DIR / "classical_confusion_matrices.png"


# -----------------------
# Feature Engineering (match your run_all.py)
# -----------------------
def feature_engineer(df: pd.DataFrame):
    df = df.copy()

    df["scam_msg_time"] = pd.to_datetime(df["scam_msg_time"], errors="coerce")
    df["first_pay_time"] = pd.to_datetime(df["first_pay_time"], errors="coerce")
    df = df.dropna(subset=["scam_msg_time", "first_pay_time"])

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["amount", "label"])

    # ---- Behavioral Features ----
    df["ipi_minutes"] = (df["first_pay_time"] - df["scam_msg_time"]).dt.total_seconds() / 60.0
    df["ipi_minutes"] = df["ipi_minutes"].clip(lower=0)

    df["response_time_minutes"] = (
        df.groupby("bank_account_no")["scam_msg_time"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
    )

    df["msg_count"] = np.clip((df["ipi_minutes"] / 15).round() + 1, 1, 12)
    df["log_amount"] = np.log1p(df["amount"])

    df["amount_z_by_account"] = (
        df.groupby("bank_account_no")["amount"]
        .transform(lambda x: (x - x.mean()) / (x.std() if x.std() else 1))
        .fillna(0)
    )

    df["ipi_z_by_account"] = (
        df.groupby("bank_account_no")["ipi_minutes"]
        .transform(lambda x: (x - x.mean()) / (x.std() if x.std() else 1))
        .fillna(0)
    )

    FEATURES = [
        "ipi_minutes",
        "response_time_minutes",
        "msg_count",
        "log_amount",
        "amount_z_by_account",
        "ipi_z_by_account",
    ]

    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], 0)
    return df, FEATURES


def sec_per_pred(model, X_test: np.ndarray) -> float:
    # quick timing: average seconds per prediction
    t0 = time.perf_counter()
    _ = model.predict(X_test)
    t1 = time.perf_counter()
    return (t1 - t0) / max(len(X_test), 1)


def eval_binary(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def plot_metrics_bar(df_res: pd.DataFrame, out_path: Path):
    labels = df_res["model"].tolist()
    metrics = ["accuracy", "precision", "recall", "f1"]

    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(9, 4.5))
    for i, m in enumerate(metrics):
        plt.bar(x + (i - 1.5) * width, df_res[m].values, width, label=m.capitalize())

    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Classical Baseline Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion_matrices(models: list[tuple[str, object]], X_test, y_test, out_path: Path):
    plt.figure(figsize=(10, 3.2))
    for i, (name, clf) in enumerate(models, start=1):
        ax = plt.subplot(1, len(models), i)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. Put this script at project root, "
            f"or edit DATA_FILE path."
        )

    print("Loading data...")
    df_raw = pd.read_csv(DATA_FILE)
    df, FEATURES = feature_engineer(df_raw)

    X = df[FEATURES].values
    y = df["label"].astype(int).values

    # same split as your pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # same scaling as your pipeline
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------
    # Train baselines
    # -----------------------
    results = []

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_metrics = eval_binary(y_test, lr_pred)
    lr_metrics["sec_per_pred"] = sec_per_pred(lr, X_test)
    results.append({"model": "Logistic Regression", **lr_metrics})

    # SVM (RBF)
    svm = SVC(kernel="rbf")
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_metrics = eval_binary(y_test, svm_pred)
    svm_metrics["sec_per_pred"] = sec_per_pred(svm, X_test)
    results.append({"model": "SVM (RBF)", **svm_metrics})

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        alpha=0.001,
        batch_size=32,
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
    )
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    mlp_metrics = eval_binary(y_test, mlp_pred)
    mlp_metrics["sec_per_pred"] = sec_per_pred(mlp, X_test)
    results.append({"model": "MLP", **mlp_metrics})

    df_res = pd.DataFrame(results)
    df_res.to_csv(CSV_OUT, index=False)
    print(f"Saved CSV: {CSV_OUT}")

    plot_metrics_bar(df_res, PNG_BAR)
    print(f"Saved plot: {PNG_BAR}")

    plot_confusion_matrices(
        [("LR", lr), ("SVM", svm), ("MLP", mlp)],
        X_test,
        y_test,
        PNG_CM,
    )
    print(f"Saved plot: {PNG_CM}")

    print("\nSummary:")
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()