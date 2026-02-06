import pandas as pd

def load_and_clean_data():
    df = pd.read_csv("data/churn.csv")

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)

    df = pd.get_dummies(df, drop_first=True)
    return df
