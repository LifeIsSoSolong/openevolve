

## Task introduction
Standalone daily pipeline for iron 2601 price forecasting using a TimeMixer model. All logic sits in `iron_future_01_daily_pipeline.py` so it can run without other project modules.
Predict 12 steps ahead.

### What the pipeline does (high level)
- Align and fuse raw series from `merged_data.csv` (or a custom CSV via `--raw_data`), keeping the target series and configured feature columns.
- Resample/fill weekly or monthly sources to daily/business-day frequency with lag rules, then forward-fill and align to the target index.
- Run feature engineering: log transform target (`y`), pct-change, rolling stats, price indicators, MACD-style indicators, supply-demand composites, and age-since-release fields; drop NaNs after feature creation.
- Build sliding-window datasets (seq_len=48 history -> pred_len=12 horizon by default), split into train/val/test by ratio, and construct time-stamp embeddings.
- Train a TimeMixer forecaster with early stopping on validation MSE; save the best weights to `checkpoints/standalone_iron_daily/best_model.pt`.
- Evaluate on the test split and report scaled MSE/MAE, value-level MAPE (after expm1), and directional accuracy.

### Data and features
- Input CSV: `merged_data.csv` with a `date` column and columns named in the fusion config (defaults embedded in the script via `DEFAULT_FUSION_CONFIG`).
- Target: `FU00002776` (log-transformed to `y`).
- Key feature groups (see `DEFAULT_FUSION_CONFIG` in the script):
  - Supply: e.g., port inventory (`ID01002312`), ship arrivals (`ID00186575`).
  - Demand/production: e.g., discharge volume (`ID00186100`), blast furnace rate (`ID00183109`).
  - Macro: e.g., PMI price (`CM0000013263`), US non-farm payrolls (`GM0000033031`).
- Feature engineering adds pct-change, rolling mean/std/slope, moving-average gaps, volatility, MACD signals, and supply-demand composite ratios/trends.

### Key hyperparameters (defaults inside `IronDailyConfig`)
- seq_len=48, pred_len=12, label_len=0, freq="b" (business day)
- batch_size=16, learning_rate=1e-2, train_epochs=10, patience=5
- TimeMixer: e_layers=4, d_layers=2, d_model=16, d_ff=32, dropout=0.1, down_sampling_layers=4, moving_avg=25, top_k=5
- Device auto-detects GPU: `cuda` if available, else `cpu`

### How to run
```bash
# from repository root
python iron_test/source_content/iron_future_01_daily_pipeline.py --raw_data iron_test/source_content/merged_data.csv
```
- Omit `--raw_data` to use the default path encoded in `DEFAULT_FUSION_CONFIG` (data/iron/merged_data.csv relative to project_root).
- Outputs: best model weights at `checkpoints/standalone_iron_daily/best_model.pt` and console metrics.

