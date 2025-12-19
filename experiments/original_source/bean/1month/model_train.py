import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import os
import joblib  # 用于保存/加载模型

# Paths
INPUT_DIR = "../data_process/brazil_train_eval"
Model_DIR = "../model_checkpoints"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SUB_PATH = "submission.csv"
MODEL_PATH = os.path.join(Model_DIR, "lgbm_model.pkl")  # 保存权重文件

# 1. Load data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# 2. Preprocessing
lag_cols = [
    "last_yield",
    "last_2_yield",
    "last_3_yield",
    "3_month_precip",
    "3_month_max",
]
train[lag_cols] = train[lag_cols].fillna(0)
test[lag_cols] = test[lag_cols].fillna(0)

train["yield"] = train.groupby(["state", "crop_year"])["yield"].transform(
    lambda x: x.fillna(x.mean())
)
train["yield"] = train["yield"].fillna(train["yield"].mean())

state2idx = {s: i for i, s in enumerate(sorted(train["state"].unique()))}
train["state_enc"] = train["state"].map(state2idx)
test["state_enc"] = test["state"].map(state2idx)

def months_since_crop_start(row):
    m = row["month"]
    if m >= 10:
        return m - 10
    else:
        return m + 2

train["months_since_crop_start"] = train.apply(months_since_crop_start, axis=1)
test["months_since_crop_start"] = test.apply(months_since_crop_start, axis=1)

feature_cols = [
    "month",
    "state_enc",
    "max",
    "min",
    "avg",
    "max-min",
    "precip",
    "3_month_precip",
    "3_month_max",
    "accumlat_precip",
    "std",
    "last_yield",
    "last_2_yield",
    "last_3_yield",
    "months_since_crop_start",
]

# 5. Cross-validation
groups = train["state"].astype(str) + "_" + train["crop_year"].astype(str)
gkf = GroupKFold(n_splits=5)
X = train[feature_cols]
y = train["yield"].values
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model = LGBMRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    rrmse = rmse / np.mean(y_val) * 100
    cv_scores.append(rrmse)
    print(f"Fold {fold+1} rRMSE: {rrmse:.3f}%")

print(f"\nMean CV rRMSE: {np.mean(cv_scores):.3f}%  (std: {np.std(cv_scores):.3f}%)")

# 6. Train on full train set
final_model = LGBMRegressor(n_estimators=300, random_state=42)
final_model.fit(X, y)

# --- 保存模型权重 ---
joblib.dump(final_model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# 7. Predict on test set
test_preds = final_model.predict(test[feature_cols])
test_preds = np.clip(test_preds, 1000, 4500)

submission = test[["year", "month", "state"]].copy()
submission["yield"] = test_preds
submission.to_csv(SUB_PATH, index=False)
print(f"Submission file saved to {SUB_PATH}")

print("\nTest prediction stats:")
print(submission["yield"].describe())

