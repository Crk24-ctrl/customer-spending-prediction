# Customer Spending Prediction

This repo trains:
- **Linear Regression** to predict `Spending Score` (numeric).
- **Logistic Regression** to predict `Spending Category` (High/Low).

Both models use one-hot encoded categorical features and evaluate on a held-out test set.

## Files
- `main.py` — end-to-end script (load → preprocess → train → evaluate → save plots).
- `train_data.csv`, `test_data.csv` — datasets (put them in the repo root).
- `requirements.txt` — Python dependencies.
- `.gitignore` — ignore local/temporary files.
- `outputs/` — created at runtime; contains results JSON + plots.

## Quickstart (Windows/macOS/Linux)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python main.py
