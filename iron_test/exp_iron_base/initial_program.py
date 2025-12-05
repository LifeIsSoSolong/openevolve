"""
Standalone pipeline for the iron_future_01_daily task.

This script aligns raw series data, applies the task-specific feature engineering
steps, builds sliding-window datasets, and trains/evaluates the TimeMixer model
end-to-end without relying on external modules from the project.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("iron_daily_pipeline")
FIX_SEED = 2021
random.seed(FIX_SEED)
np.random.seed(FIX_SEED)
torch.manual_seed(FIX_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(FIX_SEED)

# -----------------------------------------------------------------------------
# Feature engineering helpers (inlined from data_provider.feature_engineer)
# -----------------------------------------------------------------------------

class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class MillisecondOfMinute(TimeFeature):
    """Millisecond of minute encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        milliseconds = index.second * 1000 + index.microsecond // 1000
        return milliseconds / 59999.0 - 0.5


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Milli: [
            MillisecondOfMinute,
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
        ms  - milliseconds
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])

def add_age_since_release(df: pd.DataFrame, monthly_cols: List[str], date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    for col in monthly_cols:
        age_col = f"{col}_age_since_release"
        last_release_date = None
        ages = []
        for idx, (val, prev_val, cur_date) in enumerate(zip(df[col], df[col].shift(1), df[date_col])):
            if pd.isna(val):
                ages.append(np.nan)
                continue
            if idx == 0 or val != prev_val:
                last_release_date = cur_date
                ages.append(0)
            else:
                ages.append((cur_date - last_release_date).days if last_release_date else np.nan)
        df[age_col] = ages
    return df


def add_pct_change(df: pd.DataFrame, cols: List[str], periods: List[int] | None = None) -> pd.DataFrame:
    df = df.copy()
    if periods is None:
        periods = [15, 30]
    for col in cols:
        for p in periods:
            df[f"{col}_pctchg_{p}"] = df[col].pct_change(p)
    return df


def add_rolling_features_nomedian(df: pd.DataFrame, cols: List[str], windows: List[int]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        for w in windows:
            shifted = df[col].shift(1)
            df[f"{col}_rollmean_{w}"] = shifted.rolling(w).mean()
            df[f"{col}_rollstd_{w}"] = shifted.rolling(w).std()
            df[f"{col}_roll_slope{w}"] = shifted.rolling(w).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
            )
    return df


def add_price_features(df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(price_cols, list):
        price_cols = [price_cols]
    for price_col in price_cols:
        for p in [1, 3, 7]:
            df[f"{price_col}_ret_{p}d"] = df[price_col].pct_change(p)
        for w in [5, 10]:
            ma = df[price_col].rolling(w).mean()
            df[f"{price_col}_ma_{w}d"] = ma
            df[f"{price_col}_price_minus_ma_{w}d"] = df[price_col] - ma
        for v in [7, 21]:
            df[f"{price_col}_vol_{v}d"] = df[price_col].pct_change().rolling(v).std()
    return df


def add_macd_features(df: pd.DataFrame, price_col: str = "y", fast: int = 8, slow: int = 21, signal: int = 5) -> pd.DataFrame:
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    df['MACD_DIF'] = ema_fast - ema_slow
    df['MACD_DEA'] = df['MACD_DIF'].ewm(span=signal, adjust=False).mean()
    df['MACD_BAR'] = df['MACD_DIF'] - df['MACD_DEA']
    df['MACD_cross'] = (df['MACD_DIF'] > df['MACD_DEA']).astype(int)
    df['MACD_cross_above'] = ((df['MACD_DIF'] > df['MACD_DEA']) &
                              (df['MACD_DIF'].shift(1) <= df['MACD_DEA'].shift(1))).astype(int)
    df['MACD_cross_below'] = ((df['MACD_DIF'] < df['MACD_DEA']) &
                              (df['MACD_DIF'].shift(1) >= df['MACD_DEA'].shift(1))).astype(int)
    df['MACD_strength'] = df['MACD_BAR'] / df[price_col].rolling(20).mean()
    return df


def add_commodity_optimized_indicators(df: pd.DataFrame, price_col: str = 'y') -> pd.DataFrame:
    df = df.copy()
    df = add_macd_features(df, price_col=price_col, fast=8, slow=21, signal=5)
    return df


def add_supply_demand_composite_features(
    df: pd.DataFrame,
    port_inventory: str,
    supply_side: str,
    demand_side: str,
    production_activity: str,
    macro_cost: str,
) -> pd.DataFrame:
    df = df.copy()
    production_intensity = df[production_activity] * df[demand_side] / 100.0
    df['production_inventory_ratio'] = production_intensity / df[port_inventory].replace(0, np.nan)
    df['inventory_cover_days'] = df[port_inventory] / df[demand_side].replace(0, np.nan)
    df['inventory_cover_days_roll5'] = df['inventory_cover_days'].rolling(5).mean()
    df['supply_demand_gap'] = df[supply_side] - df[demand_side]
    df['supply_demand_ratio'] = df[supply_side] / df[demand_side].replace(0, np.nan)
    inventory_trend = df[port_inventory].rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    consumption_trend = df[demand_side].rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    df['inventory_structure_health'] = inventory_trend - consumption_trend
    pmi_trend = df[macro_cost].rolling(3).mean()
    consumption_trend = df[demand_side].rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    df['macro_demand_transmission'] = pmi_trend * consumption_trend
    return df


# -----------------------------------------------------------------------------
# Feature fusion helpers (derived from src/data_process/feature_fusion.py)
# -----------------------------------------------------------------------------

DEFAULT_FUSION_CONFIG = {
    "data_file": "data/iron/merged_data.csv",
    "target_name": "FU00002776",
    "output_file": "data/iron/datasets/final_features_01合约收盘价_v2.csv",
    "features": {
        "supply": [
            {
                "feature_name": "ID01002312",
                "file_path": "data/mysteel3/ID01002312_铁矿：进口：库存：45个港口（日）.csv",
                "source_column": "value",
                "fill_method": "ffill",
            },
            {
                "feature_name": "ID00186575",
                "file_path": "data/mysteel3/ID00186575_铁矿：船舶到港量：北方港口（周）.csv",
                "source_column": "value",
                "fill_method": "weekly_lag1",
            },
        ],
        "demand": [
            {
                "feature_name": "ID00186100",
                "file_path": "data/mysteel3/ID00186100_铁矿：进口：日均疏港量合计：45个港口（周）.csv",
                "source_column": "value",
                "fill_method": "weekly_lag1",
            },
            {
                "feature_name": "ID00183109",
                "file_path": "data/mysteel3/ID00183109_247家钢铁企业：高炉开工率：中国（周）.csv",
                "source_column": "value",
                "fill_method": "weekly_lag1",
            },
        ],
        "macro": [
            {
                "feature_name": "GM0000033031",
                "file_path": "data/mysteel3/GM0000033031_美国：非农就业人员：季调人数变动（月）.csv",
                "source_column": "value",
                "fill_method": "monthly_lag1_daily",
            },
            {
                "feature_name": "CM0000013263",
                "file_path": "data/mysteel3/CM0000013263_统计局：制造业PMI：购进价（月）.csv",
                "source_column": "value",
                "fill_method": "monthly_lag1_daily",
            },
        ],
    },
}


def resolve_path(base: Path, candidate: str | Path) -> Path:
    candidate_path = Path(candidate)
    if not candidate_path.is_absolute():
        candidate_path = base / candidate_path
    return candidate_path


def ensure_datetime_series(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(series.index):
        series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    return series[~series.index.duplicated(keep='last')]


def infer_weekly_rule(index: pd.DatetimeIndex, fallback: str | None = None) -> str:
    default_rule = fallback or 'W-FRI'
    if index is None or len(index) == 0:
        return default_rule
    index = pd.to_datetime(index).sort_values()
    freq = pd.infer_freq(index)
    day_map = {0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU', 4: 'FRI', 5: 'SAT', 6: 'SUN'}
    if freq and freq.startswith('W-'):
        return freq
    if freq == '7D':
        anchor_day = int(index[-1].dayofweek)
        return f"W-{day_map.get(anchor_day, 'FRI')}"
    try:
        anchor_day = int(pd.Series(index.dayofweek).mode().iloc[0])
        return f"W-{day_map.get(anchor_day, 'FRI')}"
    except Exception:
        return default_rule


def resample_with_agg(series: pd.Series, rule: str, agg: str) -> pd.Series:
    resampler = series.resample(rule, label='right', closed='right')
    agg = (agg or 'last').lower()
    if agg == 'mean':
        return resampler.mean()
    if agg == 'last':
        return resampler.last()
    if agg == 'sum':
        return resampler.sum()
    if agg == 'median':
        return resampler.median()
    raise ValueError(f"Unsupported aggregation '{agg}' for rule '{rule}'")


def apply_fill_method(
    series: pd.Series,
    method: str,
    target_index: pd.DatetimeIndex,
    weekly_rule: str,
) -> pd.Series:
    method = (method or 'ffill').lower()
    aligned_series = ensure_datetime_series(series)
    if method == 'weekly_lag1':
        weekly_series = resample_with_agg(aligned_series, weekly_rule, 'last').shift(1)
        filled = weekly_series.reindex(target_index)
        return filled.ffill()
    if method == 'monthly_lag1_daily':
        monthly_series = resample_with_agg(aligned_series, 'M', 'last').shift(1)
        filled = monthly_series.reindex(target_index)
        return filled.ffill()
    if method == 'ffill':
        aligned = aligned_series.reindex(target_index)
        filled = aligned.ffill()
        if aligned.isna().sum() > aligned_series.isna().sum():
            filled = filled.shift(1)
        return filled
    aligned = aligned_series.reindex(target_index)
    return aligned.ffill()


def build_feature_fusion_dataset(cfg: 'IronDailyConfig') -> pd.DataFrame:
    fusion_cfg = copy.deepcopy(cfg.fusion_config or DEFAULT_FUSION_CONFIG)

    data_path_str = cfg.raw_data_override or fusion_cfg.get('data_file')
    if data_path_str is None:
        raise ValueError("Fusion config must provide 'data_file'.")
    data_path = resolve_path(cfg.project_root, data_path_str)

    data_df = pd.read_csv(data_path, parse_dates=['date'])
    data_df = data_df.sort_values('date').drop_duplicates('date', keep='last')
    data_df = data_df.set_index('date')

    target_name = fusion_cfg['target_name']
    target_freq = str(fusion_cfg.get('target_frequency', 'D')).upper()
    target_agg = fusion_cfg.get('target_agg', 'last')

    target_series = ensure_datetime_series(data_df[target_name])
    weekly_rule = fusion_cfg.get('target_weekly_rule')

    if target_freq.startswith('W'):
        weekly_rule = weekly_rule or infer_weekly_rule(target_series.index)
        target_series = resample_with_agg(target_series, weekly_rule, target_agg)
    elif target_freq.startswith('M'):
        target_series = resample_with_agg(target_series, 'M', target_agg)
    target_df = target_series.dropna().to_frame(name='value')

    target_index = target_df.index
    weekly_rule = weekly_rule or infer_weekly_rule(target_index)

    final_df = target_df.copy()
    feature_groups = fusion_cfg.get('features', {})
    for group_features in feature_groups.values():
        for feature in group_features:
            feature_name = feature['feature_name']
            fill_method = feature.get('fill_method', 'ffill')
            if feature_name not in data_df.columns:
                raise KeyError(f"Feature '{feature_name}' not found in raw dataset.")
            series = data_df[feature_name]
            processed = apply_fill_method(series, fill_method, target_index, weekly_rule)
            final_df[feature_name] = processed

    final_df = final_df.sort_index().ffill().dropna()
    final_df = final_df.reset_index().rename(columns={'index': 'date'})

    return final_df


# -----------------------------------------------------------------------------
# TimeMixer implementation (inlined from models/TimeMixer.py)
# -----------------------------------------------------------------------------


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DFTSeriesDecomp(nn.Module):
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xf = torch.fft.rfft(x)
        freq = torch.abs(xf)
        freq[..., 0] = 0
        top_k_freq, _ = torch.topk(freq, self.top_k)
        xf = torch.where(freq > top_k_freq.min(), xf, torch.zeros_like(xf))
        x_season = torch.fft.irfft(xf, n=x.size(1))
        x_trend = x - x_season
        return x_season, x_trend


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular',
            bias=False,
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model: int, embed_type: str = 'fixed', freq: str = 'h'):
        super().__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        embed_cls = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = embed_cls(minute_size, d_model)
        if freq in ['t', 'h']:
            self.hour_embed = embed_cls(hour_size, d_model)
        self.weekday_embed = embed_cls(weekday_size, d_model)
        self.day_embed = embed_cls(day_size, d_model)
        self.month_embed = embed_cls(month_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3]) if hasattr(self, 'hour_embed') else 0.
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        w = torch.zeros(c_in, d_model).float()
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x).detach()


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model: int, freq: str = 'h'):
        super().__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'ms': 7, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        self.embed = nn.Linear(freq_map[freq], d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class DataEmbeddingWoPos(nn.Module):
    def __init__(self, c_in: int, d_model: int, embed_type: str, freq: str, dropout: float):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        if embed_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
        else:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor | None, x_mark: torch.Tensor | None) -> torch.Tensor:
        if x is None and x_mark is not None:
            return self.temporal_embedding(x_mark)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, non_norm: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.non_norm = non_norm
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self.mean = None
        self.stdev = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            if not self.non_norm:
                dims = tuple(range(1, x.ndim - 1))
                self.mean = torch.mean(x, dim=dims, keepdim=True).detach()
                self.stdev = torch.sqrt(torch.var(x, dim=dims, keepdim=True, unbiased=False) + self.eps).detach()
                x = (x - self.mean) / self.stdev
                if self.affine:
                    x = x * self.affine_weight + self.affine_bias
            return x
        if mode == 'denorm':
            if not self.non_norm and self.mean is not None and self.stdev is not None:
                if self.affine:
                    x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
                x = x * self.stdev + self.mean
            return x
        raise NotImplementedError


class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                ),
                nn.GELU(),
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                ),
            )
            for i in range(configs.down_sampling_layers)
        ])

    def forward(self, season_list: List[torch.Tensor]) -> List[torch.Tensor]:
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))
        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    configs.seq_len // (configs.down_sampling_window ** i),
                ),
                nn.GELU(),
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                ),
            )
            for i in reversed(range(configs.down_sampling_layers))
        ])

    def forward(self, trend_list: List[torch.Tensor]) -> List[torch.Tensor]:
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))
        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence
        if configs.decomp_method == 'moving_avg':
            self.decomposition = SeriesDecomp(configs.moving_avg)
        elif configs.decomp_method == 'dft_decomp':
            self.decomposition = DFTSeriesDecomp(configs.top_k)
        else:
            raise ValueError('Unsupported decomposition method')
        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU(),
                nn.Linear(configs.d_ff, configs.d_model),
            )
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model),
        )

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        length_list = [x.size(1) for x in x_list]
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs) for _ in range(configs.e_layers)])
        self.preprocess = SeriesDecomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature
        self.future_gate = nn.Linear(2 * configs.d_model, configs.d_model) if self.use_future_temporal_feature else None
        self.dir_adjust_scale = getattr(configs, 'dir_adjust_scale', 20)
        if self.channel_independence == 1:
            self.enc_embedding = DataEmbeddingWoPos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbeddingWoPos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.normalize_layers = nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
            for _ in range(configs.down_sampling_layers + 1)
        ])
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.predict_layers = nn.ModuleList([
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ])
            dir_out_channels = 1 if self.channel_independence == 1 else configs.c_out
            self.direction_head = nn.Linear(configs.d_model, dir_out_channels, bias=True)
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)
                self.out_res_layers = nn.ModuleList([
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])
                self.regression_layers = nn.ModuleList([
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])
        elif self.task_name in ['imputation', 'anomaly_detection']:
            out_dim = 1 if self.channel_independence == 1 else configs.c_out
            self.projection_layer = nn.Linear(configs.d_model, out_dim, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)
        else:
            raise ValueError('Unsupported task name')

    def out_projection(self, dec_out: torch.Tensor, i: int, out_res: torch.Tensor) -> torch.Tensor:
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        return dec_out + out_res

    def pre_enc(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor] | None]:
        if self.channel_independence == 1:
            return x_list, None
        out1_list, out2_list = [], []
        for x in x_list:
            x_1, x_2 = self.preprocess(x)
            out1_list.append(x_1)
            out2_list.append(x_2)
        return out1_list, out2_list

    def __multi_scale_process_inputs(
        self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor] | None]:
        if self.configs.down_sampling_method == 'max':
            down_pool = nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(
                in_channels=self.configs.enc_in,
                out_channels=self.configs.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.configs.down_sampling_window,
            )
        else:
            raise ValueError('Unknown down sampling method')

        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list: List[torch.Tensor] = []
        x_mark_sampling_list: List[torch.Tensor] | None = None
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        if x_mark_enc is not None:
            x_mark_sampling_list = [x_mark_enc]

        for _ in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None and x_mark_sampling_list is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None and x_mark_sampling_list is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc
        return x_enc, x_mark_enc

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
    ) -> torch.Tensor:
        self.future_time_embed = None
        if self.use_future_temporal_feature and x_mark_dec is not None:
            B, _, N = x_enc.size()
            future_mark = x_mark_dec[:, -self.pred_len:, :]
            if self.channel_independence == 1:
                future_mark = future_mark.repeat(N, 1, 1)
            self.future_time_embed = self.enc_embedding(None, future_mark)
        x_enc_list, x_mark_list = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        x_list = []
        x_mark_processed = []
        if x_mark_list is not None:
            for x, x_mark, norm_layer in zip(x_enc_list, x_mark_list, self.normalize_layers):
                x = norm_layer(x, 'norm')
                if self.channel_independence == 1:
                    B, T, N = x.size()
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_processed.append(x_mark)
        else:
            for x, norm_layer in zip(x_enc_list, self.normalize_layers):
                x = norm_layer(x, 'norm')
                if self.channel_independence == 1:
                    B, T, N = x.size()
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
        enc_out_list = []
        processed = self.pre_enc(x_list)
        if self.channel_independence == 1:
            processed_list = processed
            if x_mark_list is not None:
                for x, x_mark in zip(processed_list, x_mark_processed):
                    enc_out_list.append(self.enc_embedding(x, x_mark))
            else:
                for x in processed_list:
                    enc_out_list.append(self.enc_embedding(x, None))
        else:
            enc_inputs, out_res_list = processed
            if x_mark_list is not None:
                for x, x_mark in zip(enc_inputs, x_mark_processed):
                    enc_out_list.append(self.enc_embedding(x, x_mark))
            else:
                for x in enc_inputs:
                    enc_out_list.append(self.enc_embedding(x, None))
            x_list = (enc_inputs, out_res_list)
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
        dec_out_list = self.future_multi_mixing(x_enc.size(0), enc_out_list, x_list)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B: int, enc_out_list: List[torch.Tensor], x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                if self.use_future_temporal_feature and self.future_time_embed is not None:
                    fusion = torch.cat([dec_out, self.future_time_embed], dim=-1)
                    gate = torch.sigmoid(self.future_gate(fusion))
                    dec_out = dec_out + gate * (self.future_time_embed - dec_out)
                dir_logits = self.direction_head(dec_out)
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dir_logits = dir_logits.reshape(B, 1, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)
        else:
            enc_inputs, out_res_list = x_list
            for i, (enc_out, out_res) in enumerate(zip(enc_out_list, out_res_list)):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                if self.use_future_temporal_feature and self.future_time_embed is not None:
                    fusion = torch.cat([dec_out, self.future_time_embed], dim=-1)
                    gate = torch.sigmoid(self.future_gate(fusion))
                    dec_out = dec_out + gate * (self.future_time_embed - dec_out)
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)
        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.task_name == 'imputation':
            raise NotImplementedError('Imputation path is not required for this script')
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError('Anomaly detection path is not required for this script')
        if self.task_name == 'classification':
            raise NotImplementedError('Classification path is not required for this script')
        raise ValueError('Unknown task')


# -----------------------------------------------------------------------------
# Pipeline configuration and training logic
# -----------------------------------------------------------------------------


@dataclass
class IronDailyConfig:
    # Use current working directory as the base so the script can be run from anywhere
    project_root: Path = Path.cwd()
    checkpoint_dir: Path | None = None
    raw_data_override: str | None = None
    fusion_config: Dict[str, Any] | None = None
    label_len: int = 0
    pred_len: int = 12
    freq: str = "b"
    target_col: str = "y"
    seq_len: int = 48
    batch_size: int = 16
    learning_rate: float = 1e-2
    train_epochs: int = 10
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    e_layers: int = 4
    d_layers: int = 2
    d_model: int = 16
    d_ff: int = 32
    dropout: float = 0.1
    down_sampling_layers: int = 4
    down_sampling_window: int = 2
    factor: int = 1
    channel_independence: int = 0
    c_out: int = 1
    use_future_temporal_feature: int = 0
    moving_avg: int = 25
    decomp_method: str = "moving_avg"
    top_k: int = 5
    embed: str = "timeF"
    use_norm: int = 1
    dir_adjust_scale: float = 20.0
    split_ratio: Dict[str, float] | None = None
    def __post_init__(self) -> None:
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_root / "checkpoints" / "standalone_iron_daily"
        if self.fusion_config is None:
            self.fusion_config = copy.deepcopy(DEFAULT_FUSION_CONFIG)
        if self.split_ratio is None:
            self.split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def device_obj(self) -> torch.device:
        return torch.device(self.device)


def fuse_and_align_features(cfg: 'IronDailyConfig') -> pd.DataFrame:
    fusion_df = build_feature_fusion_dataset(cfg)
    fusion_df["date"] = pd.to_datetime(fusion_df["date"])
    fusion_df = fusion_df.sort_values("date").reset_index(drop=True)
    return fusion_df


def run_feature_engineering(df: pd.DataFrame, cfg: IronDailyConfig) -> pd.DataFrame:
    df = df.copy()
    df["y"] = np.log1p(df["value"])
    cols = list(df.columns)
    cols.remove(cfg.target_col)
    remove_list = ["value", "contract_id", "date"] + [f"value_lag_{i + 1}" for i in range(4, 10)]
    cols = [c for c in cols if c not in remove_list]
    df = df[["date"] + cols + [cfg.target_col]]
    df = add_age_since_release(df, monthly_cols=["GM0000033031"], date_col="date")
    df = add_pct_change(df, cols=["ID00186575", "ID00186100"])
    df = add_rolling_features_nomedian(df, cols=["ID01002312"], windows=[3, 5, 15])
    df = add_price_features(df, price_cols=["ID00183109"])
    df = add_commodity_optimized_indicators(df, price_col="y")
    df = add_supply_demand_composite_features(
        df,
        port_inventory="ID01002312",
        supply_side="ID00186575",
        demand_side="ID00186100",
        production_activity="ID00183109",
        macro_cost="CM0000013263",
    )
    df = df.dropna().reset_index(drop=True)
    return df


def compute_split_borders(total_len: int, cfg: IronDailyConfig) -> Tuple[List[int], List[int]]:
    ratios = cfg.split_ratio
    train_ratio = float(ratios.get("train", 0.8))
    val_ratio = float(ratios.get("val", 0.1))
    test_ratio = float(ratios.get("test", 0.1))
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio = 1.0 - train_ratio - val_ratio

    num_train = int(total_len * train_ratio)
    num_val = int(total_len * val_ratio)
    num_test = total_len - num_train - num_val
    if num_train <= 0 or num_test <= 0:
        raise ValueError("Insufficient data after applying split ratios.")

    border1s = [0, max(num_train - cfg.seq_len, 0), total_len - num_test - cfg.seq_len]
    border2s = [num_train, num_train + num_val, total_len]
    return border1s, border2s


def build_time_mark_array(dates: pd.Series, cfg: IronDailyConfig) -> np.ndarray:
    if cfg.embed == 'timeF':
        date_array = pd.to_datetime(dates.values)
        data_stamp = time_features(date_array, freq=cfg.freq)
        return data_stamp.transpose(1, 0)
    df_stamp = pd.DataFrame({'date': pd.to_datetime(dates)})
    df_stamp['month'] = df_stamp['date'].dt.month
    df_stamp['day'] = df_stamp['date'].dt.day
    df_stamp['weekday'] = df_stamp['date'].dt.weekday
    df_stamp['hour'] = df_stamp['date'].dt.hour
    return df_stamp[['month', 'day', 'weekday', 'hour']].values


def prepare_custom_style_data(df: pd.DataFrame, cfg: IronDailyConfig):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.assign(**{cfg.target_col: df.pop(cfg.target_col)})
    feature_cols = [c for c in df.columns if c != 'date']
    data_values = df[feature_cols].values.astype(np.float32)
    total_len = len(df)
    border1s, border2s = compute_split_borders(total_len, cfg)
    split_info = {}
    names = ['train', 'val', 'test']
    for idx, name in enumerate(names):
        b1, b2 = border1s[idx], border2s[idx]
        data_slice = data_values[b1:b2]
        stamp_slice = build_time_mark_array(df['date'].iloc[b1:b2], cfg)
        split_info[name] = {
            'data': data_slice,
            'stamp': stamp_slice.astype(np.float32),
            'length': len(data_slice),
            'dates': df['date'].iloc[b1:b2].to_numpy(),
        }
    return split_info, feature_cols


class CustomStyleDataset(Dataset):
    def __init__(self, data: np.ndarray, stamp: np.ndarray, seq_len: int, label_len: int,
                 pred_len: int, set_type: int, stride_test: int, dates: np.ndarray):
        self.data_x = torch.from_numpy(data)
        self.data_y = torch.from_numpy(data)
        self.data_stamp = torch.from_numpy(stamp)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.set_type = set_type
        self.stride_test = stride_test
        self.dates = dates

    def __len__(self) -> int:
        total_windows = len(self.data_x) - self.seq_len - self.pred_len + 1
        if total_windows <= 0:
            return 0
        if self.set_type == 2:
            return max(total_windows // self.stride_test, 0)
        return total_windows

    def _calc_indices(self, idx: int):
        stride = self.stride_test if self.set_type == 2 else 1
        max_s_begin = len(self.data_x) - self.seq_len - self.pred_len
        s_begin = max_s_begin - idx * stride
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return s_begin, s_end, r_begin, r_end

    def __getitem__(self, idx: int):
        s_begin, s_end, r_begin, r_end = self._calc_indices(idx)
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x.float(), seq_y.float(), seq_x_mark.float(), seq_y_mark.float()

    def window_bounds(self, idx: int):
        s_begin, s_end, _, _ = self._calc_indices(idx)
        start_date = pd.Timestamp(self.dates[s_begin])
        end_date = pd.Timestamp(self.dates[s_end - 1])
        return start_date, end_date


def make_dataloaders_from_splits(
    split_info: Dict[str, Dict[str, np.ndarray]], cfg: IronDailyConfig
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    freq = cfg.freq.lower()
    stride_test = 2 if freq.startswith('m') else 12
    set_types = {'train': 0, 'val': 1, 'test': 2}
    for split_name, set_type in set_types.items():
        entry = split_info[split_name]
        dataset = CustomStyleDataset(
            entry['data'],
            entry['stamp'],
            cfg.seq_len,
            cfg.label_len,
            cfg.pred_len,
            set_type,
            stride_test,
            entry['dates'],
        )
        batch_size = cfg.batch_size if split_name != 'test' else 1
        shuffle = split_name == 'train'
        loaders[split_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return loaders


def build_model(cfg: IronDailyConfig, enc_in: int) -> TimeMixer:
    model_args = {
        "task_name": "long_term_forecast",
        "seq_len": cfg.seq_len,
        "label_len": cfg.label_len,
        "pred_len": cfg.pred_len,
        "down_sampling_window": cfg.down_sampling_window,
        "down_sampling_layers": cfg.down_sampling_layers,
        "channel_independence": cfg.channel_independence,
        "e_layers": cfg.e_layers,
        "d_layers": cfg.d_layers,
        "moving_avg": cfg.moving_avg,
        "use_future_temporal_feature": cfg.use_future_temporal_feature,
        "d_model": cfg.d_model,
        "d_ff": cfg.d_ff,
        "dropout": cfg.dropout,
        "embed": cfg.embed,
        "freq": cfg.freq,
        "enc_in": enc_in,
        "dec_in": enc_in,
        "c_out": cfg.c_out,
        "factor": cfg.factor,
        "use_norm": cfg.use_norm,
        "decomp_method": cfg.decomp_method,
        "top_k": cfg.top_k,
        "dir_adjust_scale": cfg.dir_adjust_scale,
        "down_sampling_method": "avg",
    }
    model_cfg = SimpleNamespace(**model_args)
    return TimeMixer(model_cfg)


def extract_target(pred: torch.Tensor, batch_y: torch.Tensor, cfg: IronDailyConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    f_dim = -1 if cfg.c_out == 1 else 0
    pred_y = pred[:, -cfg.pred_len :, f_dim:]
    true_y = batch_y[:, -cfg.pred_len :, f_dim:]
    return pred_y, true_y


def compute_directional_accuracy(pred_value: np.ndarray, true_value: np.ndarray) -> float:
    if pred_value.shape[1] < 2:
        return float("nan")
    pred_diff = np.diff(pred_value, axis=1)
    true_diff = np.diff(true_value, axis=1)
    agreement = np.sign(pred_diff) == np.sign(true_diff)
    return float(np.mean(agreement))


def evaluate(
    model: TimeMixer,
    loader: DataLoader,
    cfg: IronDailyConfig,
    device: torch.device,
    apply_log_transform: bool = True,
) -> Tuple[float, float, float, float]:
    model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)
            if cfg.down_sampling_layers == 0:
                dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :cfg.label_len, :], dec_inp], dim=1).to(device)
            else:
                dec_inp = None
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred_y, true_y = extract_target(outputs, batch_y, cfg)
            preds.append(pred_y.cpu().numpy())
            trues.append(true_y.cpu().numpy())
    preds_arr = np.concatenate(preds, axis=0)
    trues_arr = np.concatenate(trues, axis=0)
    scaled_mse = np.mean((preds_arr - trues_arr) ** 2)
    scaled_mae = np.mean(np.abs(preds_arr - trues_arr))
    if apply_log_transform:
        pred_value = np.expm1(preds_arr)
        true_value = np.expm1(trues_arr)
    else:
        pred_value = preds_arr
        true_value = trues_arr
    value_mape = np.mean(np.abs((pred_value - true_value) / np.clip(true_value, 1e-6, None)))
    da_score = compute_directional_accuracy(pred_value, true_value)
    return scaled_mse, scaled_mae, value_mape, da_score


def train_pipeline(cfg: IronDailyConfig) -> None:
    print("1) 数据对齐：对原始序列进行工作日频率重采样并填充...")
    fused_df = fuse_and_align_features(cfg)
    print(f"   对齐后样本数: {len(fused_df)}")

    print("2) 特征工程：复用日频任务所需的所有变换...")
    fe_df = run_feature_engineering(fused_df, cfg)
    print(f"   特征工程完成，剩余样本: {len(fe_df)}")

    print("3) 数据集切分与标准化...")
    split_info, feature_cols = prepare_custom_style_data(fe_df, cfg)
    enc_in = len(feature_cols)
    print(f"   输入特征维度 enc_in={enc_in}")
    loaders = make_dataloaders_from_splits(split_info, cfg)
    dataset_sizes = {split: len(loader.dataset) for split, loader in loaders.items()}
    loader_steps = {split: len(loader) for split, loader in loaders.items()}
    logger.info(
        "Dataset windows -> train:%d, val:%d, test:%d",
        dataset_sizes.get("train", 0),
        dataset_sizes.get("val", 0),
        dataset_sizes.get("test", 0),
    )
    print(
        f"   数据窗口数量：train={dataset_sizes.get('train', 0)}, "
        f"val={dataset_sizes.get('val', 0)}, test={dataset_sizes.get('test', 0)}"
    )
    logger.info(
        "Loader steps/epoch -> train:%d, val:%d, test:%d",
        loader_steps.get("train", 0),
        loader_steps.get("val", 0),
        loader_steps.get("test", 0),
    )
    print(
        f"   Dataloader步数：train={loader_steps.get('train', 0)}, "
        f"val={loader_steps.get('val', 0)}, test={loader_steps.get('test', 0)}"
    )
    test_dataset = loaders["test"].dataset
    print("   Test窗口时间跨度：")
    for idx in range(len(test_dataset)):
        start_date, end_date = test_dataset.window_bounds(idx)
        print(f"     波段{idx + 1:02d}: {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")

    print("4) 模型初始化与训练...")
    model = build_model(cfg, enc_in).to(cfg.device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    logger.info(
        "Training params | epochs=%d, batch=%d, lr=%.4f, patience=%d, seq_len=%d, pred_len=%d, d_model=%d, d_ff=%d",
        cfg.train_epochs,
        cfg.batch_size,
        cfg.learning_rate,
        cfg.patience,
        cfg.seq_len,
        cfg.pred_len,
        cfg.d_model,
        cfg.d_ff,
    )
    print(
        f"   训练参数：epochs={cfg.train_epochs}, batch={cfg.batch_size}, lr={cfg.learning_rate}, "
        f"patience={cfg.patience}, seq_len={cfg.seq_len}, pred_len={cfg.pred_len}, "
        f"d_model={cfg.d_model}, d_ff={cfg.d_ff}"
    )
    logger.info(
        "Model depth | e_layers=%d, d_layers=%d, down_sampling_layers=%d, down_window=%d",
        cfg.e_layers,
        cfg.d_layers,
        cfg.down_sampling_layers,
        cfg.down_sampling_window,
    )
    print(
        f"   模型结构：e_layers={cfg.e_layers}, d_layers={cfg.d_layers}, "
        f"down_layers={cfg.down_sampling_layers}, down_window={cfg.down_sampling_window}"
    )
    best_val = math.inf
    best_state = None
    patience_counter = 0
    for epoch in range(cfg.train_epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loaders["train"]:
            batch_x = batch_x.to(cfg.device_obj)
            batch_y = batch_y.to(cfg.device_obj)
            batch_x_mark = batch_x_mark.to(cfg.device_obj)
            batch_y_mark = batch_y_mark.to(cfg.device_obj)
            if cfg.down_sampling_layers == 0:
                dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :cfg.label_len, :], dec_inp], dim=1).to(cfg.device_obj)
            else:
                dec_inp = None
            optimizer.zero_grad()
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred_y, true_y = extract_target(outputs, batch_y, cfg)
            loss = criterion(pred_y, true_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / max(len(loaders["train"]), 1)
        val_mse, _, _, _ = evaluate(model, loaders["val"], cfg, cfg.device_obj)
        print(f"   Epoch {epoch + 1:02d}: train_loss={avg_loss:.4f}, val_mse={val_mse:.4f}")
        if val_mse < best_val:
            best_val = val_mse
            best_state = model.state_dict()
            patience_counter = 0
            logger.info("New best validation MSE %.6f at epoch %d", val_mse, epoch + 1)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print("   早停：验证集未提升。")
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), cfg.checkpoint_dir / "best_model.pt")

    print("5) 测试集评估...")
    test_mse, test_mae, test_mape, test_da = evaluate(
        model, loaders["test"], cfg, cfg.device_obj
    )
    print(
        f"   Test metrics -> scaled_MSE: {test_mse:.4f}, scaled_MAE: {test_mae:.4f}, "
        f"value_MAPE: {test_mape:.4f}, DA: {test_da:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone iron_future_01_daily pipeline")
    parser.add_argument(
        "--raw_data",
        type=str,
        default=None,
        help="Path to the merged raw dataset (overrides config data_file)",
    )
    args = parser.parse_args()

    configuration = IronDailyConfig(raw_data_override=args.raw_data)
    train_pipeline(configuration)
    
    # 运行命令：uv run src/test_task/iron_future_01_daily_pipeline.py --raw_data data/iron/merged_data.csv
