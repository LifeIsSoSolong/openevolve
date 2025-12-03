import os
import pandas as pd
import lightgbm as lgb

# =========================
# 路径配置
# =========================
INPUT_DIR = "./temp_data/input"
OUTPUT_DIR = "./temp_data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

# =========================
# 工具函数：州编码 + 月份重编码
# =========================
def encode_state(df, mapping=None):
    """对 state 列进行数值编码"""
    df = df.copy()
    if mapping is None:
        states = sorted(df["state"].unique())
        mapping = {s: i for i, s in enumerate(states)}
    df["state_enc"] = df["state"].map(mapping).fillna(-1).astype(int)
    return df, mapping

def months_since_crop_start(df):
    """根据月份生成生育期编码"""
    df = df.copy()
    def transform(m):
        return m - 10 if m >= 10 else m + 2
    df["months_since_crop_start"] = df["month"].apply(transform)
    return df

# =========================
# 主流程
# =========================
def main():
    # ---------- 读取 ----------
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # ---------- 州编码 & 月份重编码 ----------
    train, state2idx = encode_state(train)
    test, _ = encode_state(test, mapping=state2idx)
    train = months_since_crop_start(train)
    test = months_since_crop_start(test)

    # ---------- 特征选择 ----------
    # 训练时，使用数值特征；避免 LightGBM 因 object 列报错
    numeric_kinds = ("b", "i", "u", "f", "c")
    candidate_features = [col for col in train.columns if col not in ["yield"]]
    features = [
        col for col in candidate_features
        if train[col].dtype.kind in numeric_kinds
    ]
    target = "yield"

    # ---------- 训练 ----------
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(train[features], train[target])

    # ---------- 预测 ----------
    test_pred = model.predict(test[features])

    # ---------- 输出 ----------
    test_out = test.copy()
    test_out["yield"] = test_pred
    test_out = test_out[["year", "month", "state", "yield"]]  # 按要求输出四列

    # 保存结果
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    test_out.to_csv(out_path, index=False)
    print(f"✅ 模型训练完成，预测结果已保存至: {out_path}")

if __name__ == "__main__":
    main()
