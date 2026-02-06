import numpy as np
import pandas as pd

def load_and_clean_data():
    df = pd.read_csv("data/churn.csv")

    # Clean numeric column safely
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing critical values
    df = df.dropna(subset=["TotalCharges"])

    # Drop ID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    # Replace any inf/-inf created by weird values (rare, but prevents sklearn warnings)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df
