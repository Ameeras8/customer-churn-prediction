# Customer Churn Prediction

End-to-end customer churn prediction project built using a **professional local ML workflow** (VS Code, virtual environments, modular Python scripts, GitHub).

**Key highlights**
- Built and trained Logistic Regression and Random Forest models
- Achieved ~80% accuracy on real-world telco churn data
- Identified top churn drivers using feature importance
- Fully reproducible, local-first ML pipeline (no Colab)


# Customer Churn Prediction (VS Code + Local ML Pipeline)

Predict customer churn using a clean, professional ML workflow (local development in VS Code, modular Python scripts, Git versioning).

## Project Structure
- `src/download_data.py` — downloads dataset and saves locally
- `src/preprocess.py` — cleans + encodes features
- `src/train.py` — trains Logistic Regression + Random Forest and prints evaluation
- `src/feature_importance.py` — identifies top churn drivers and saves a plot

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Main churn signals: tenure, total charges, monthly charges, fiber optic service, electronic check payments

Business action: target short-tenure + high-monthly-charge customers with contract offers + support bundles