# results_experiments.py
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from classical_models import train_lr, train_svm, train_mlp
from qnn_model import train_qnn

from make_results_png import make_results_png
from plot_qnn_analysis import plot_qnn_all_metrics_vs_qubits, plot_qnn_vs_mlp
from plot_qnn_learning_curve import plot_loss_vs_steps


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATA_FILE = PROJECT_ROOT / "transactions_2000.csv"

RESULTS_CSV = RESULTS_DIR / "results_all.csv"
RESULTS_PNG = RESULTS_DIR / "results_all.png"


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

    df = df.sort_values(["bank_account_no", "scam_msg_time"]).reset_index(drop=True)

    df["ipi_minutes"] = (df["first_pay_time"] - df["scam_msg_time"]).dt.total_seconds() / 60.0
    df["ipi_minutes"] = df["ipi_minutes"].clip(lower=0).fillna(df["ipi_minutes"].median() if not df["ipi_minutes"].median() != df["ipi_minutes"].median() else 0.0)

    df["prev_scam_msg_time"] = df.groupby("bank_account_no")["scam_msg_time"].shift(1)
    df["response_time_minutes"] = (df["scam_msg_time"] - df["prev_scam_msg_time"]).dt.total_seconds() / 60.0
    df["response_time_minutes"] = df["response_time_minutes"].fillna(df["response_time_minutes"].median() if not df["response_time_minutes"].median() != df["response_time_minutes"].median() else 0.0).clip(lower=0)

    rng = np.random.default_rng(42)
    base = np.clip(df["ipi_minutes"].to_numpy(), 0, 120)
    msg_count = (1 + (base / 15.0)).round().astype(int)
    msg_count = msg_count + rng.integers(0, 2, size=len(df))
    df["msg_count"] = np.clip(msg_count, 1, 12)

    df["log_amount"] = np.log1p(df["amount"].clip(lower=0))

    grp_mean = df.groupby("bank_account_no")["amount"].transform("mean")
    grp_std = df.groupby("bank_account_no")["amount"].transform("std").replace(0, np.nan)
    df["amount_z_by_account"] = ((df["amount"] - grp_mean) / grp_std).fillna(0.0)

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


def adjust_dim(X: np.ndarray, n_qubits: int) -> np.ndarray:
    d = X.shape[1]
    if n_qubits == d:
        return X
    if n_qubits < d:
        return X[:, :n_qubits]
    pad = np.zeros((X.shape[0], n_qubits - d), dtype=float)
    return np.hstack([X, pad])


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find dataset: {DATA_FILE}")

    df = feature_engineer(pd.read_csv(DATA_FILE))

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results_rows = []

    # -------- Classical baselines --------
    lr_out = train_lr(X_train_s, y_train, X_test_s, y_test)
    results_rows.append({"model": "Logistic Regression", **{k: lr_out[k] for k in ["accuracy","precision","recall","f1","sec_per_pred"]}})

    svm_out = train_svm(X_train_s, y_train, X_test_s, y_test)
    results_rows.append({"model": "SVM (RBF)", **{k: svm_out[k] for k in ["accuracy","precision","recall","f1","sec_per_pred"]}})

    mlp_out = train_mlp(X_train_s, y_train, X_test_s, y_test, seed=42)
    results_rows.append({"model": "Neural Network (MLP)", **{k: mlp_out[k] for k in ["accuracy","precision","recall","f1","sec_per_pred"]}})

    # -------- QNN: 4/5/6 qubits --------
    qubits_list = [4, 5, 6]
    qnn_acc, qnn_prec, qnn_rec, qnn_f1 = [], [], [], []
    qnn_best = None
    qnn_best_qubits = None

    for q in qubits_list:
        out = train_qnn(
            adjust_dim(X_train_s, q), y_train,
            adjust_dim(X_test_s, q), y_test,
            n_qubits=q,
            n_layers=3,
            steps=400,
            lr=0.05,
            batch_size=16,
            encoding="dense",
            optimizer_name="adam",
            shots=None
        )

        # loss curve
        if out.get("loss_history"):
            plot_loss_vs_steps(out["loss_history"], RESULTS_DIR / f"qnn_loss_curve_{q}q.png")

        results_rows.append({
            "model": f"QNN ({q} qubits)",
            "accuracy": out["accuracy"],
            "precision": out["precision"],
            "recall": out["recall"],
            "f1": out["f1"],
            "sec_per_pred": out["sec_per_pred"],
        })

        qnn_acc.append(out["accuracy"])
        qnn_prec.append(out["precision"])
        qnn_rec.append(out["recall"])
        qnn_f1.append(out["f1"])

        # choose "best" QNN by F1 (or change to accuracy if you prefer)
        if (qnn_best is None) or (out["f1"] > qnn_best["f1"]):
            qnn_best = out
            qnn_best_qubits = q

    # -------- Save table + combined results figure --------
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(RESULTS_CSV, index=False)

    make_results_png(results_df, RESULTS_PNG)

    # -------- QNN metrics vs qubits --------
    plot_qnn_all_metrics_vs_qubits(
        qubits=qubits_list,
        accuracy=qnn_acc,
        precision=qnn_prec,
        recall=qnn_rec,
        f1=qnn_f1,
        output_path=str(RESULTS_DIR / "qnn_all_metrics_vs_qubits.png")
    )

    # -------- QNN vs MLP --------
    if qnn_best is not None:
        plot_qnn_vs_mlp(
            mlp_metrics=[mlp_out["accuracy"], mlp_out["precision"], mlp_out["recall"], mlp_out["f1"]],
            qnn_metrics=[qnn_best["accuracy"], qnn_best["precision"], qnn_best["recall"], qnn_best["f1"]],
            output_path=str(RESULTS_DIR / f"qnn_vs_mlp_best_{qnn_best_qubits}q.png")
        )

    print("\n✅ All results generated in:", RESULTS_DIR)
    print("✅ CSV:", RESULTS_CSV)
    print("✅ Main figure:", RESULTS_PNG)


if __name__ == "__main__":
    main()