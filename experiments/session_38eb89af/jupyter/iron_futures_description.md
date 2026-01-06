# 铁矿石期货价格预测 - MLE 任务描述

## 任务类型
时间序列回归 - 多步预测

## 数据说明

### 数据文件
- **文件名**: `data_iron_2601_8w.csv`（或其他铁矿石期货数据文件）/ 提前拆分好的train, test数据
- **数据格式**: CSV 表格数据
- **时间范围**: 周级别时间序列数据
- **数据列**:
  - `date`: 日期列（周级别）
  - `value`: 目标变量 - 铁矿石期货周收盘价（人民币/吨）
  - 其他列: 相关特征指标（ID系列、GM/CM编码等）

### 数据划分
- **训练集/测试集比例**: 80/20（按时间顺序划分）
- **划分方式**: 时间序列顺序划分，不打乱
- **保存要求**: 划分后保存为 `train.csv` 和 `test.csv`

## 预测任务

### 任务目标
使用历史数据预测**未来 8 周**的铁矿石期货周收盘价

### 方法论
- **滑动窗口方法**:
  - 输入：历史 M 周的数据（M 是可优化参数，建议初始值 8-16）
  - 输出：未来固定 8 周的价格预测
  - 样本构建：从原始时间序列中滑动窗口生成多个 (M+8) 长度的样本

### 特征工程
- **时间特征**: date 列应提取为时间特征（年、月、季度、周、周期性编码等）
- **其他特征**: 所有数值列作为特征，需处理缺失值
- **可优化方向**: 滞后特征、移动平均、趋势特征等

## 评估指标

### 主要指标（权重 60%）
- **MDA (Mean Directional Accuracy)**: 方向准确率
  - 定义：预测趋势方向（涨/跌）与实际方向的一致性
  - 范围：0-100%
  - 目标：**最大化**

### 次要指标（各权重 20%）
- **RMSE (Root Mean Squared Error)**: 均方根误差
  - 目标：最小化
  
- **MAPE (Mean Absolute Percentage Error)**: 平均绝对百分比误差
  - 范围：0-100%
  - 目标：最小化

## 模型要求

### 必须满足
1. **单模型多输出**: 使用 1 个模型同时预测 8 周数据（不能是 8 个独立模型）
2. **数据泄露防范**: 
   - 不能使用未来信息
   - 测试集不参与训练
   - 统计量只从训练集计算
3. **预测窗口固定**: 输出必须是 8 周

### 推荐模型
- GradientBoostingRegressor + MultiOutputRegressor
- XGBoost + MultiOutputRegressor
- LightGBM + MultiOutputRegressor
- RandomForest (原生支持多输出)
- 深度学习模型（如 MLP, LSTM, Transformer）

## 优化方向

### EVOLVE-BLOCK 应包含的可优化部分

1. **窗口大小优化**
   ```python
   def get_window_size() -> int:
       return 12  # 可优化：尝试 4-24 周
   ```

2. **时间特征工程**
   ```python
   def extract_time_features(df):
       # 可优化：年月季度、周期性编码、滞后特征、移动平均等
   ```

3. **缺失值处理**
   ```python
   def preprocess_features(df):
       # 可优化：中位数、均值、前向填充、插值等
   ```

4. **滑动窗口样本构建**
   ```python
   def create_sliding_window_samples(df, window_size, horizon):
       # 可优化：特征展平方式、归一化等
   ```

5. **模型构建**
   ```python
   def build_model():
       # 可优化：模型类型、超参数
   ```

6. **评估指标计算**
   ```python
   def compute_mda(y_true, y_pred):
       # MDA 计算逻辑
   
   def compute_metrics(y_true, y_pred):
       # 返回 {"mda": ..., "rmse": ..., "mape": ...}
   ```

## AlphaEvolve 配置建议

- **进化轮数**: 5-10 轮
- **评估超时**: 3000 秒（数据量小可以缩短到 1800 秒）
- **种群大小**: 16
- **进化追踪**: 启用（用于生成进化树可视化）

## 约束条件

1. **时间限制**: 单次评估应在合理时间内完成（建议 < 10 分钟）
2. **内存限制**: 注意滑动窗口样本数量，避免内存溢出
3. **数据格式**: 确保 agent.py 的 main(root) 返回格式为:
   ```python
   return {
       "mda": float,    # 0-100
       "rmse": float,   # > 0
       "mape": float    # 0-100
   }
   ```

## 使用方式

### 文件准备
1. 准备数据文件（如 `data_iron_2601_8w.csv`）
2. 准备本 description.md 文件

### 调用 skill
上传文件到 `$EVO_INPUT_DIR`，然后告诉 Claude/DeepAgents:

```
使用 alphaevolve-mle-code-creator skill，根据 description.md 和数据文件创建 MLE 代码
```

### 生成内容
skill 将自动生成：
- `agent.py`: 符合 AlphaEvolve 规范的代码
- `train.csv`, `test.csv`: 划分后的数据
- 与你交互确认任务细节和 EVOLVE-BLOCK 范围

---

## 注意事项

1. **数据质量**: 确保数据按时间排序，无重大缺失
2. **特征含义**: 如有特殊业务含义的特征，在交互时说明
3. **指标权重**: 可根据业务需求调整 MDA/RMSE/MAPE 的权重比例
4. **baseline**: 首次运行会建立 baseline，后续进化会尝试超越
