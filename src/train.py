import pandas as pd
from preprocess import load_and_clean_data

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    df = load_and_clean_data()

    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]

    # Only scale continuous numeric columns (professional best practice)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    # If any are missing for some reason, keep only those that exist
    numeric_cols = [c for c in numeric_cols if c in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="passthrough",  # keep one-hot columns as-is
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr_model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )

    rf_model = RandomForestClassifier(n_estimators=300, random_state=42)

    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    print("\n=== Logistic Regression (scaled numeric only) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))

    print("\n=== Random Forest ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))


if __name__ == "__main__":
    main()
