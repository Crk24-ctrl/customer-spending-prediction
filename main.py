import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def read_csvs(train_csv: str, test_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not Path(train_csv).exists():
        raise FileNotFoundError(f"Could not find train CSV at: {train_csv}")
    if not Path(test_csv).exists():
        raise FileNotFoundError(f"Could not find test CSV at: {test_csv}")

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    return train, test


def one_hot_encode_and_align(train: pd.DataFrame, test: pd.DataFrame,
                             categorical_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode train and test separately, then align columns
    so both have the exact same feature set (missing columns filled with 0).
    """
    train_enc = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
    test_enc = pd.get_dummies(test, columns=categorical_cols, drop_first=True)

    # Align columns to train's columns
    missing_in_test = set(train_enc.columns) - set(test_enc.columns)
    for col in missing_in_test:
        test_enc[col] = 0

    missing_in_train = set(test_enc.columns) - set(train_enc.columns)
    for col in missing_in_train:
        train_enc[col] = 0

    # Ensure same column order
    test_enc = test_enc[train_enc.columns]

    return train_enc, test_enc


def split_features_labels_for_linear(train_enc: pd.DataFrame, test_enc: pd.DataFrame,
                                     label_col: str,
                                     leakage_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    For regression:
      - y = Spending Score
      - X = all columns except label and any leakage columns (e.g., Spending Category)
    """
    if label_col not in train_enc.columns or label_col not in test_enc.columns:
        raise KeyError(f"Missing label column '{label_col}' in encoded dataframes.")

    X_train = train_enc.drop(columns=set(leakage_cols + [label_col]) & set(train_enc.columns))
    X_test = test_enc.drop(columns=set(leakage_cols + [label_col]) & set(test_enc.columns))
    y_train = train_enc[label_col]
    y_test = test_enc[label_col]
    return X_train, y_train, X_test, y_test


def split_features_labels_for_logistic(train_enc: pd.DataFrame, test_enc: pd.DataFrame,
                                       label_col: str,
                                       drop_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    For classification:
      - y = Spending Category (string labels are OK)
      - X = all columns except Spending Category + columns to drop (e.g., Spending Score)
    """
    if label_col not in train_enc.columns or label_col not in test_enc.columns:
        raise KeyError(f"Missing label column '{label_col}' in encoded dataframes.")

    X_train = train_enc.drop(columns=set(drop_cols + [label_col]) & set(train_enc.columns))
    X_test = test_enc.drop(columns=set(drop_cols + [label_col]) & set(test_enc.columns))
    y_train = train_enc[label_col]
    y_test = test_enc[label_col]
    return X_train, y_train, X_test, y_test


def ensure_outdir(outdir: str) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_true_vs_pred(y_true: pd.Series, y_pred: np.ndarray, outpath: Path):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, label="Predictions")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, linestyle="--", linewidth=1, label="Ideal")
    plt.xlabel("True Spending Score")
    plt.ylabel("Predicted Spending Score")
    plt.title("Linear Regression — True vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_confusion(y_true: pd.Series, y_pred: np.ndarray, outpath: Path):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    plt.figure(figsize=(6.5, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Logistic Regression — Confusion Matrix")
    plt.xticks(ticks=range(len(np.unique(y_true))), labels=np.unique(y_true), rotation=45)
    plt.yticks(ticks=range(len(np.unique(y_true))), labels=np.unique(y_true))
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train linear and logistic models for customer spending data.")
    parser.add_argument("--train_csv", default="train_data.csv")
    parser.add_argument("--test_csv", default="test_data.csv")
    parser.add_argument("--outdir", default="outputs")
    # change these if your dataset uses different headers
    parser.add_argument("--spending_score_col", default="Spending Score")
    parser.add_argument("--spending_category_col", default="Spending Category")
    parser.add_argument("--categorical_cols", nargs="*", default=["Gender", "Profession"])
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)

    # 1) Load
    train, test = read_csvs(args.train_csv, args.test_csv)

    # 2) One-hot encode + align
    train_enc, test_enc = one_hot_encode_and_align(
        train.copy(),
        test.copy(),
        categorical_cols=args.categorical_cols
    )

    # ---- Linear Regression (predict Spending Score) ----
    Xtr_lin, ytr_lin, Xte_lin, yte_lin = split_features_labels_for_linear(
        train_enc, test_enc,
        label_col=args.spending_score_col,
        leakage_cols=[args.spending_category_col]  # drop Category to avoid leakage
    )
    lin_model = LinearRegression()
    lin_model.fit(Xtr_lin, ytr_lin)
    ypred_lin = lin_model.predict(Xte_lin)
    mse = float(mean_squared_error(yte_lin, ypred_lin))

    plot_true_vs_pred(yte_lin, ypred_lin, outdir / "linear_true_vs_pred.png")
    with open(outdir / "linear_results.json", "w", encoding="utf-8") as f:
        json.dump({"mse": mse}, f, indent=2)

    # ---- Logistic Regression (predict Spending Category) ----
    Xtr_log, ytr_log, Xte_log, yte_log = split_features_labels_for_logistic(
        train_enc, test_enc,
        label_col=args.spending_category_col,
        drop_cols=[args.spending_score_col]  # do not use Score to predict Category
    )

    # Scale numeric features helps convergence; we’ll scale all features (ok for linear models)
    scaler = StandardScaler(with_mean=False)  # with_mean=False keeps sparse safety if any
    Xtr_log_s = scaler.fit_transform(Xtr_log)
    Xte_log_s = scaler.transform(Xte_log)

    log_model = LogisticRegression(
        solver="liblinear",     # robust on small datasets, supports L1/L2
        penalty="l2",
        max_iter=2000,
        n_jobs=None
    )
    log_model.fit(Xtr_log_s, ytr_log)
    ypred_log = log_model.predict(Xte_log_s)
    acc = float(accuracy_score(yte_log, ypred_log))

    # Class balance info (handy to interpret accuracy)
    class_counts = yte_log.value_counts().to_dict()
    with open(outdir / "logistic_results.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "class_counts_test": class_counts}, f, indent=2)

    plot_confusion(yte_log, ypred_log, outdir / "logistic_confusion_matrix.png")

    # Console summary
    print("\n=== Results ===")
    print(f"Linear Regression  -> MSE: {mse:.4f}")
    print(f"Logistic Regression-> Accuracy: {acc:.4f}")
    print(f"\nSaved plots and JSONs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
