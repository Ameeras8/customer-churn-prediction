import pandas as pd
import matplotlib.pyplot as plt
from preprocess import load_and_clean_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    df = load_and_clean_data()
    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)

    print("\nTop 15 Churn Drivers (Random Forest):")
    print(importances)

    plt.figure()
    importances.sort_values().plot(kind="barh")
    plt.title("Top 15 Churn Drivers (Random Forest)")
    plt.tight_layout()
    plt.savefig("churn_feature_importance.png", dpi=200)
    print("\n✅ Saved plot: churn_feature_importance.png")

if __name__ == "__main__":
    main()
