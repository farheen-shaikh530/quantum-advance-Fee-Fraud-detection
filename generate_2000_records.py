# generate_2000_records.py
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
IN_FILE  = PROJECT_ROOT / "transactions.csv"          # your original file
OUT_FILE = PROJECT_ROOT / "transactions_2000.csv"

TARGET_N = 2000
SEED = 42

def main():
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(IN_FILE)

    # Make sure required columns exist
    required = ["bank_account_no", "scam_msg_time", "first_pay_time", "amount", "label"]
    for c in required:
        if c not in df.columns:
            df[c] = ""

    # Bootstrap sampling: sample rows WITH replacement to reach 2000
    df_big = df.sample(n=TARGET_N, replace=True, random_state=SEED).copy()

    # Optional: add small perturbations so duplicates aren’t identical
    # amount jitter
    df_big["amount"] = pd.to_numeric(df_big["amount"], errors="coerce").fillna(0.0)
    df_big["amount"] = (df_big["amount"] + rng.normal(0, 2.0, size=len(df_big))).clip(lower=0)

    # Ensure account ids exist (helps your groupby features)
    df_big["bank_account_no"] = df_big["bank_account_no"].fillna("").astype(str).str.strip()
    missing = (df_big["bank_account_no"] == "") | (df_big["bank_account_no"].str.lower() == "nan")
    if missing.any():
        df_big.loc[missing, "bank_account_no"] = [f"acct_{i%200}" for i in range(missing.sum())]

    df_big.to_csv(OUT_FILE, index=False)
    print(f"✅ Wrote: {OUT_FILE}  (rows={len(df_big)})")

if __name__ == "__main__":
    main()