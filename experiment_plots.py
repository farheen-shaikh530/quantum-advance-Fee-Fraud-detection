# experiment_plots.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from classical_models import train_mlp
from qnn_model import train_qnn


# -----------------------
# Paths
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATA_FILE = PROJECT_ROOT / "transactions_2000.csv"   # <-- change if needed

OUT_QUBIT_CSV = RESULTS_DIR / "qnn_qubits_metrics.csv"
OUT_QUBIT_PNG = RESULTS_DIR / "qnn_metrics_vs_qubits.png"

OUT_QNN_MLP_CSV = RESULTS_DIR / "qnn_vs_mlp.csv"
OUT_QNN_MLP_PNG = RESULTS_DIR / "qnn_vs_mlp.png"


# -----------------------
# Feature engineering (same as your run_all.py)
# -----------------------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["scam_msg_time"] = pd.to_datetime(df["scam_msg_time"], errors="coerce")
    df["first_pay_time"] = pd.to_datetime(df["first_pay_time"], errors="coerce")
    df = df.dropna(subset=["scam_msg_time", "first_pay_time"]).copy()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["amount", "label"]).copy()

    if "bank_account_no" not in df.columns:
        df["bank_account_no"] = ""

    df["bank_account_no"] = df["bank_account_no"].fillna("").astype(str).str.strip()
    missing_mask = (df["bank_account_no"] == "") | (df["bank_account_no"].str.lower() == "nan")
    if missing_mask.any():
        df.loc[missing_mask, "bank_account_no"] = df.loc[missing_mask].index.map(lambda i: f"acct_{i % 10}")

    df = df.sort_values(["bank_account_no", "scam_msg_time"]).reset_index(drop=True)

    # F1: IPI
    df["ipi_minutes"] = (df["first_pay_time"] - df["scam_msg_time"]).dt.total_seconds() / 60.0
    df["ipi_minutes"] = df["ipi_minutes"].clip(lower=0)
    ipi_med = df["ipi_minutes"].median()
    if pd.isna(ipi_med):
        ipi_med = 0.0
    df["ipi_minutes"] = df["ipi_minutes"].fillna(ipi_med)

    # F2: response time
    df["prev_scam_msg_time"] = df.groupby("bank_account_no")["scam_msg_time"].shift(1)
    df["response_time_minutes"] = (df["scam_msg_time"] - df["prev_scam_msg_time"]).dt.total_seconds() / 60.0
    med = df["response_time_minutes"].median()
    if pd.isna(med):
        med = 0.0
    df["response_time_minutes"] = df["response_time_minutes"].fillna(med).clip(lower=0)

    # F3: msg_count proxy
    rng = np.random.default_rng(42)
    base = np.clip(df["ipi_minutes"].to_numpy(), 0, 120)
    msg_count = (1 + (base / 15.0)).round().astype(int)
    msg_count = msg_count + rng.integers(0, 2, size=len(df))
    df["msg_count"] = np.clip(msg_count, 1, 12)

    # F4: log amount
    df["log_amount"] = np.log1p(df["amount"].clip(lower=0))

    # F5: amount z by account
    grp_mean = df.groupby("bank_account_no")["amount"].transform("mean")
    grp_std = df.groupby("bank_account_no")["amount"].transform("std").replace(0, np.nan)
    df["amount_z_by_account"] = ((df["amount"] - grp_mean) / grp_std).fillna(0.0)

    # F6: ipi z by account
    grp_mean_ipi = df.groupby("bank_account_no")["ipi_minutes"].transform("mean")
    grp_std_ipi = df.groupby("bank_account_no")["ipi_minutes"].transform("std").replace(0, np.nan)
    df["ipi_z_by_account"] = ((df["ipi_minutes"] - grp_mean_ipi) / grp_std_ipi).fillna(0.0)

    FEATURE_COLS = [
        "ipi_minutes",
        "response_time_minutes",
        "msg_count",
        "log_amount",
        "amount_z_by_account",
        "ipi_z_by_account",
    ]
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# -----------------------
# Helper: match feature dims to qubit count
# - if qubits < 6: use first qubits features
# - if qubits > 6: pad with zeros (extra qubits carry no extra info)
# -----------------------
def adjust_dim(X: np.ndarray, n_qubits: int) -> np.ndarray:
    d = X.shape[1]
    if n_qubits == d:
        return X
    if n_qubits < d:
        return X[:, :n_qubits]
    pad = np.zeros((X.shape[0], n_qubits - d), dtype=float)
    return np.hstack([X, pad])


def plot_metrics_vs_qubits(df_metrics: pd.DataFrame, out_png: Path):
    # Line plot: Accuracy/Precision/Recall/F1 vs #qubits
    qubits = df_metrics["qubits"].to_numpy()

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(qubits, df_metrics["accuracy"], marker="o", label="Accuracy")
    plt.plot(qubits, df_metrics["precision"], marker="o", label="Precision")
    plt.plot(qubits, df_metrics["recall"], marker="o", label="Recall")
    plt.plot(qubits, df_metrics["f1"], marker="o", label="F1-score")

    plt.title("QNN Metrics vs Number of Qubits")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Metric Value")
    plt.xticks(qubits)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_qnn_vs_mlp_bar(df_cmp: pd.DataFrame, out_png: Path):
    # Bar chart: compare QNN vs MLP across metrics
    models = df_cmp["model"].tolist()
    metrics = ["accuracy", "precision", "recall", "f1"]

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(9, 4.8))
    for i, m in enumerate(metrics):
        plt.bar(x + (i - 1.5) * width, df_cmp[m].to_numpy(), width, label=m.capitalize())

    plt.title("QNN vs Classical Neural Network (MLP): Metric Comparison")
    plt.xlabel("Model")
    plt.ylabel("Metric Value")
    plt.xticks(x, models, rotation=10)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find: {DATA_FILE}")

    df_raw = pd.read_csv(DATA_FILE)
    df = feature_engineer(df_raw)

    if df["label"].nunique() < 2:
        raise ValueError(f"Need at least 2 label classes. Found: {df['label'].unique()}")

    FEATURE_COLS = [
        "ipi_minutes",
        "response_time_minutes",
        "msg_count",
        "log_amount",
        "amount_z_by_account",
        "ipi_z_by_account",
    ]

    X = df[FEATURE_COLS].astype(float).values
    y = df["label"].astype(int).values

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale in [0,1] (fit train only)
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # -----------------------------
    # (1) QNN metrics vs #qubits
    # -----------------------------
    qubit_list = [4, 6, 8]
    rows = []
    for q in qubit_list:
        Xtr_q = adjust_dim(X_train_s, q)
        Xte_q = adjust_dim(X_test_s, q)

        out = train_qnn(
            Xtr_q, y_train, Xte_q, y_test,
            n_qubits=q,
            n_layers=3,
            steps=1000,
            lr=0.03,
            batch_size=16
        )
        rows.append({
            "qubits": q,
            "accuracy": out["accuracy"],
            "precision": out["precision"],
            "recall": out["recall"],
            "f1": out["f1"],
            "sec_per_pred": out["sec_per_pred"],
        })

    df_qubits = pd.DataFrame(rows).sort_values("qubits")
    df_qubits.to_csv(OUT_QUBIT_CSV, index=False)
    plot_metrics_vs_qubits(df_qubits, OUT_QUBIT_PNG)

    print(f"\n✅ Saved QNN qubit sweep CSV: {OUT_QUBIT_CSV}")
    print(f"✅ Saved QNN qubit sweep plot: {OUT_QUBIT_PNG}")
    print(df_qubits.to_string(index=False))

    # -----------------------------
    # (2) QNN vs MLP comparison
    # - Use 6 features/qubits for fairness
    # -----------------------------
    mlp_out = train_mlp(X_train_s, y_train, X_test_s, y_test, seed=42)

    qnn6_out = train_qnn(
        adjust_dim(X_train_s, 6), y_train,
        adjust_dim(X_test_s, 6), y_test,
        n_qubits=6,
        n_layers=3,
        steps=1000,
        lr=0.03,
        batch_size=16
    )

    df_cmp = pd.DataFrame([
        {
            "model": "Neural Network (MLP, 6F)",
            "accuracy": mlp_out["accuracy"],
            "precision": mlp_out["precision"],
            "recall": mlp_out["recall"],
            "f1": mlp_out["f1"],
            "sec_per_pred": mlp_out["sec_per_pred"],
        },
        {
            "model": "QNN (6 qubits, 6F)",
            "accuracy": qnn6_out["accuracy"],
            "precision": qnn6_out["precision"],
            "recall": qnn6_out["recall"],
            "f1": qnn6_out["f1"],
            "sec_per_pred": qnn6_out["sec_per_pred"],
        },
    ])
    df_cmp.to_csv(OUT_QNN_MLP_CSV, index=False)
    plot_qnn_vs_mlp_bar(df_cmp, OUT_QNN_MLP_PNG)

    print(f"\n✅ Saved QNN vs MLP CSV: {OUT_QNN_MLP_CSV}")
    print(f"✅ Saved QNN vs MLP plot: {OUT_QNN_MLP_PNG}")
    print(df_cmp.to_string(index=False))


if __name__ == "__main__":
    main()