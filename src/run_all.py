# run_all.py
# ============================================================
# Main experiment runner for:
# - Classical ML baselines
# - QNN qubit scaling (4,5,6,7)
# - Plots for results section
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .classical_models import train_lr, train_svm, train_mlp
from .qnn_model import train_qnn
from .make_results_png import make_results_png
# -----------------------
# Paths
# -----------------------
ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATA_FILE = ROOT / "transactions_2000.csv"

CSV_OUT = RESULTS_DIR / "results_all.csv"
PNG_MODEL = RESULTS_DIR / "model_comparison.png"
PNG_QUBITS = RESULTS_DIR / "qnn_metrics_vs_qubits.png"


# -----------------------
# Feature Engineering
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


# -----------------------
# Utility: match qubits
# -----------------------
def adjust_dim(X: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    If n_qubits > #features, pad with zeros.
    If n_qubits < #features, truncate.
    """
    if X.shape[1] >= n_qubits:
        return X[:, :n_qubits]
    pad = np.zeros((X.shape[0], n_qubits - X.shape[1]))
    return np.hstack([X, pad])


# -----------------------
# Plot helpers
# -----------------------
def plot_qnn_vs_qubits(out_path: Path, qubits, acc, prec, rec, f1):
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(qubits, acc, "o-", label="Accuracy")
    plt.plot(qubits, prec, "o-", label="Precision")
    plt.plot(qubits, rec, "o-", label="Recall")
    plt.plot(qubits, f1, "o-", label="F1-score")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Metric Value")
    plt.title("QNN Performance vs Number of Qubits")
    plt.xticks(qubits)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_qnn_vs_mlp(out_path: Path, mlp_metrics, qnn_metrics, qnn_label: str):
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7.2, 4.5))
    plt.bar(x - width / 2, mlp_metrics, width, label="MLP")
    plt.bar(x + width / 2, qnn_metrics, width, label=qnn_label)

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Metric Value")
    plt.title("QNN vs Classical Neural Network")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_loss_curve(out_path: Path, loss_history, title: str):
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------
# Main Experiment
# -----------------------
def main():
    print("Loading data...")
    df_raw = pd.read_csv(DATA_FILE)
    df, FEATURES = feature_engineer(df_raw)

    X = df[FEATURES].values
    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = []

    # -----------------------
    # Classical Models
    # -----------------------
    print("Training classical models...")
    lr = train_lr(X_train, y_train, X_test, y_test)
    svm = train_svm(X_train, y_train, X_test, y_test)
    mlp = train_mlp(X_train, y_train, X_test, y_test)
    
    results.append({**lr, "model": "Logistic Regression"})
    results.append({**svm, "model": "SVM"})
    results.append({**mlp, "model": "Neural Network"})




    # Store MLP metrics for later plotting
    mlp_metrics = [mlp["accuracy"], mlp["precision"], mlp["recall"], mlp["f1"]]

    # -----------------------
    # QNN: Qubit Scaling (4,5,6,7)
    # -----------------------
    print("Training QNNs (qubit sweep)...")

    qubits = [5]
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    qnn_by_qubits = {}

    for q in qubits:
        out = train_qnn(
            adjust_dim(X_train, q), y_train,
            adjust_dim(X_test, q), y_test,
            n_qubits=q,
            n_layers=3,
            steps=400,
            lr=0.05,
            batch_size=16,
            encoding="dense",
            optimizer_name="adam"
        )

        qnn_by_qubits[q] = out

        # collect for qubits-vs-metrics plot
        acc_list.append(out["accuracy"])
        prec_list.append(out["precision"])
        rec_list.append(out["recall"])
        f1_list.append(out["f1"])
        
        results.append({**out, "model": f"QNN ({q} qubits)"})

        # loss plot if available
        if "loss_history" in out and out["loss_history"] is not None:
            plot_loss_curve(
                RESULTS_DIR / f"loss_qnn_{q}q.png",
                out["loss_history"],
                title=f"QNN Training Loss ({q} qubits)"
            )

    # -----------------------
    # Save outputs
    # -----------------------
    df_res = pd.DataFrame(results)
    df_res.to_csv(CSV_OUT, index=False)
    print(f"Saved: {CSV_OUT}")

    # main 3-panel figure (accuracy + PRF + latency)
    make_results_png(df_res, PNG_MODEL)

    # QNN metrics vs qubits
    plot_qnn_vs_qubits(PNG_QUBITS, qubits, acc_list, prec_list, rec_list, f1_list)

    # QNN vs MLP (6q and 7q)
    if 6 in qnn_by_qubits:
        q6 = qnn_by_qubits[6]
        qnn6_metrics = [q6["accuracy"], q6["precision"], q6["recall"], q6["f1"]]
        plot_qnn_vs_mlp(
            RESULTS_DIR / "qnn_vs_mlp_6q.png",
            mlp_metrics,
            qnn6_metrics,
            qnn_label="QNN (6 qubits)"
        )

    if 7 in qnn_by_qubits:
        q7 = qnn_by_qubits[7]
        qnn7_metrics = [q7["accuracy"], q7["precision"], q7["recall"], q7["f1"]]
        plot_qnn_vs_mlp(
            RESULTS_DIR / "qnn_vs_mlp_7q.png",
            mlp_metrics,
            qnn7_metrics,
            qnn_label="QNN (7 qubits)"
        )

    print("All figures generated in /results")


if __name__ == "__main__":
    main()