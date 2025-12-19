# run_all.py
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from plot_qnn_learning_curve import plot_loss_vs_steps

from classical_models import train_lr, train_svm, train_mlp
from qnn_model import train_qnn

from make_results_png import make_results_png
from plot_qnn_analysis import (
    plot_qnn_all_metrics_vs_qubits,
    plot_qnn_vs_mlp
)

# -----------------------
# Paths (always project-root)
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_CSV = RESULTS_DIR / "results_all.csv"
RESULTS_PNG = RESULTS_DIR / "results_all.png"

DATA_FILE = PROJECT_ROOT / "transactions_2000.csv"  # change if needed


# -----------------------
# Feature engineering (6F)
# -----------------------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse timestamps
    df["scam_msg_time"] = pd.to_datetime(df["scam_msg_time"], errors="coerce")
    df["first_pay_time"] = pd.to_datetime(df["first_pay_time"], errors="coerce")

    # Drop rows with bad times
    df = df.dropna(subset=["scam_msg_time", "first_pay_time"]).copy()

    # Numeric cleanup
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["amount", "label"]).copy()

    # bank_account_no handling
    if "bank_account_no" not in df.columns:
        df["bank_account_no"] = ""

    df["bank_account_no"] = df["bank_account_no"].fillna("").astype(str).str.strip()
    missing_mask = (df["bank_account_no"] == "") | (df["bank_account_no"].str.lower() == "nan")
    if missing_mask.any():
        # pseudo accounts for per-account group features
        df.loc[missing_mask, "bank_account_no"] = df.loc[missing_mask].index.map(lambda i: f"acct_{i % 10}")

    # Sort for sequential features
    df = df.sort_values(["bank_account_no", "scam_msg_time"]).reset_index(drop=True)

    # -----------------------
    # Feature 1: ipi_minutes
    # -----------------------
    df["ipi_minutes"] = (df["first_pay_time"] - df["scam_msg_time"]).dt.total_seconds() / 60.0
    df["ipi_minutes"] = df["ipi_minutes"].clip(lower=0)
    ipi_med = df["ipi_minutes"].median()
    df["ipi_minutes"] = df["ipi_minutes"].fillna(0.0 if pd.isna(ipi_med) else ipi_med)

    # -----------------------
    # Feature 2: response_time_minutes (proxy)
    # -----------------------
    df["prev_scam_msg_time"] = df.groupby("bank_account_no")["scam_msg_time"].shift(1)
    df["response_time_minutes"] = (df["scam_msg_time"] - df["prev_scam_msg_time"]).dt.total_seconds() / 60.0
    med = df["response_time_minutes"].median()
    df["response_time_minutes"] = df["response_time_minutes"].fillna(0.0 if pd.isna(med) else med).clip(lower=0)

    # -----------------------
    # Feature 3: msg_count (proxy)
    # -----------------------
    rng = np.random.default_rng(42)
    base = np.clip(df["ipi_minutes"].to_numpy(), 0, 120)
    msg_count = (1 + (base / 15.0)).round().astype(int)
    msg_count = msg_count + rng.integers(0, 2, size=len(df))
    df["msg_count"] = np.clip(msg_count, 1, 12)

    # -----------------------
    # Feature 4: log_amount
    # -----------------------
    df["log_amount"] = np.log1p(df["amount"].clip(lower=0))

    # -----------------------
    # Feature 5: amount_z_by_account
    # -----------------------
    grp_mean = df.groupby("bank_account_no")["amount"].transform("mean")
    grp_std = df.groupby("bank_account_no")["amount"].transform("std").replace(0, np.nan)
    df["amount_z_by_account"] = ((df["amount"] - grp_mean) / grp_std).fillna(0.0)

    # -----------------------
    # Feature 6: ipi_z_by_account
    # -----------------------
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

    # Final safety: replace inf and fill NaNs
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find: {DATA_FILE}")

    df_raw = pd.read_csv(DATA_FILE)
    df = feature_engineer(df_raw)

    # Safety checks
    if len(df) < 10:
        raise ValueError(f"Not enough valid rows after preprocessing. Rows left: {len(df)}")
    if df["label"].nunique() < 2:
        raise ValueError(f"Need at least 2 classes in label. Found: {df['label'].unique()}")

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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results_rows = []

    # -----------------------
    # Classical models
    # -----------------------
    lr_out = train_lr(X_train_s, y_train, X_test_s, y_test)
    results_rows.append({"model": "Logistic Regression (6F)", **{k: lr_out[k] for k in ["accuracy","precision","recall","f1","sec_per_pred"]}})

    svm_out = train_svm(X_train_s, y_train, X_test_s, y_test)
    results_rows.append({"model": "SVM RBF (6F)", **{k: svm_out[k] for k in ["accuracy","precision","recall","f1","sec_per_pred"]}})

    mlp_out = train_mlp(X_train_s, y_train, X_test_s, y_test, seed=42)
    results_rows.append({"model": "Neural Network (MLP, 6F)", **{k: mlp_out[k] for k in ["accuracy","precision","recall","f1","sec_per_pred"]}})

    # -----------------------
    # QNN scaling (4, 5, 6)
    # -----------------------
   # -----------------------
    # QNN scaling (4, 5, 6)
    # -----------------------
    qubits_list = [4, 5, 6]
    qnn_acc_list, qnn_prec_list, qnn_rec_list, qnn_f1_list = [], [], [], []
    qnn6_for_compare = None

    for nq in qubits_list:
        qnn_out = train_qnn(
            X_train_s, y_train, X_test_s, y_test,
            n_qubits=nq,
            n_layers=3,
            steps=400,
            lr=0.05,
            batch_size=16,
            encoding="dense",
            optimizer_name="adam",
            shots=None
        )

        # ✅ Save loss curve PNG (requires train_qnn to return "loss_history")
        if "loss_history" in qnn_out and qnn_out["loss_history"] is not None:
            plot_loss_vs_steps(
                qnn_out["loss_history"],
                RESULTS_DIR / f"qnn_loss_curve_{nq}q.png"
            )

        # Save row metrics (DON'T store the full loss history in CSV)
        results_rows.append({
            "model": f"QNN {nq}-Qubit (6F)",
            "accuracy": qnn_out["accuracy"],
            "precision": qnn_out["precision"],
            "recall": qnn_out["recall"],
            "f1": qnn_out["f1"],
            "sec_per_pred": qnn_out["sec_per_pred"],
        })

        # collect for qubit-scaling plot
        qnn_acc_list.append(qnn_out["accuracy"])
        qnn_prec_list.append(qnn_out["precision"])
        qnn_rec_list.append(qnn_out["recall"])
        qnn_f1_list.append(qnn_out["f1"])

        if nq == 6:
            qnn6_for_compare = qnn_out
    # -----------------------
    # Save CSV + main bar plot
    # -----------------------
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(RESULTS_CSV, index=False)

    print(f"\n✅ Saved CSV: {RESULTS_CSV}")
    print(results_df.to_string(index=False))

    make_results_png(results_df, RESULTS_PNG)
    print(f"\n✅ Saved PNG: {RESULTS_PNG}")

    # -----------------------
    # QNN-only scaling plot (Accuracy/Precision/Recall/F1 vs qubits)
    # -----------------------
    plot_qnn_all_metrics_vs_qubits(
        qubits=qubits_list,
        accuracy=qnn_acc_list,
        precision=qnn_prec_list,
        recall=qnn_rec_list,
        f1=qnn_f1_list,
        output_path=str(RESULTS_DIR / "qnn_all_metrics_vs_qubits.png")
    )

    # -----------------------
    # QNN vs MLP plot
    # -----------------------
    if qnn6_for_compare is not None:
        plot_qnn_vs_mlp(
            mlp_metrics=[mlp_out["accuracy"], mlp_out["precision"], mlp_out["recall"], mlp_out["f1"]],
            qnn_metrics=[qnn6_for_compare["accuracy"], qnn6_for_compare["precision"], qnn6_for_compare["recall"], qnn6_for_compare["f1"]],
            output_path=str(RESULTS_DIR / "qnn_vs_mlp.png")
        )


if __name__ == "__main__":
    main()