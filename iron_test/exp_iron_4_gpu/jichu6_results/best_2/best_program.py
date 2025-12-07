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
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import os

# Ensure deterministic CuBLAS workspace for CUDA deterministic algorithms
if torch.cuda.is_available():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("iron_daily_pipeline")
FIX_SEED = 2021

def _set_global_seed(seed: int = 2021) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch versions may not have this or certain ops may not support it
            pass

_set_global_seed(FIX_SEED)

# Generator for DataLoader to keep shuffling deterministic
_shared_generator = torch.Generator()
_shared_generator.manual_seed(FIX_SEED)

def _worker_init_fn(worker_id: int) -> None:
    # Ensure each worker has a deterministic seed derived from global seed
    worker_seed = FIX_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# EVOLVE-BLOCK-START

# -----------------------------------------------------------------------------
# Feature engineering helpers (inlined from data_provider.feature_engineer)
# -----------------------------------------------------------------------------

def time_features(dates, freq: str = "b") -> np.ndarray:
    """Business-day calendar features (dow/dom/doy scaled to [-0.5, 0.5])."""
    dates = pd.to_datetime(dates)
    dow = dates.dayofweek / 6.0 - 0.5
    dom = (dates.day - 1) / 30.0 - 0.5
    doy = (dates.dayofyear - 1) / 365.0 - 0.5
    return np.vstack([dow, dom, doy])

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


# Feature fusion is disabled in this standalone script; keep a tiny placeholder.
DEFAULT_FUSION_CONFIG: Dict[str, Any] = {}


def build_feature_fusion_dataset(cfg: 'IronDailyConfig') -> pd.DataFrame:  # pragma: no cover
    raise NotImplementedError(
        "Feature fusion is disabled; provide cached train_raw/val_raw/test_raw CSVs instead."
    )


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


# DFTSeriesDecomp is unnecessary here because decomp_method is fixed to 'moving_avg'.


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


# TemporalEmbedding / FixedEmbedding stubs are not needed since embed='timeF'
# always routes through TimeFeatureEmbedding in DataEmbeddingWoPos.


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
        # For this task we always use calendar time features (embed='timeF')
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
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
        self.channel_independence = configs.channel_independence
        if configs.decomp_method != 'moving_avg':
            raise ValueError('Unsupported decomposition method')
        self.decomposition = SeriesDecomp(configs.moving_avg)
        if self.channel_independence == 0:
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
        season_list: List[torch.Tensor] = []
        trend_list: List[torch.Tensor] = []
        for x in x_list:
            season, trend = self.decomposition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        out_list: List[torch.Tensor] = []
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
            # Learnable weights for aggregating multi-scale predictions instead of a simple sum
            self.scale_weights = nn.Parameter(torch.ones(configs.down_sampling_layers + 1))
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
        # In this task we always set use_future_temporal_feature=0, so we skip
        # the unused future-time gating logic and directly build multi-scale
        # encoder inputs. This keeps the forward pass compact but is behaviour-
        # equivalent for the current configuration.
        x_enc_list, x_mark_list = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        x_list: List[torch.Tensor] = []
        x_mark_processed: List[torch.Tensor] = []
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
        enc_out_list: List[torch.Tensor] = []
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
        # Multi-scale regression + projection
        dec_out_list = self.future_multi_mixing(x_enc.size(0), enc_out_list, x_list)
        dec_out_stack = torch.stack(dec_out_list, dim=-1)
        # Aggregate predictions from different scales using learnable softmax weights
        if hasattr(self, "scale_weights"):
            weights = torch.softmax(self.scale_weights, dim=0)
            dec_out = (dec_out_stack * weights.view(1, 1, 1, -1)).sum(-1)
        else:
            dec_out = dec_out_stack.sum(-1)
        # Denormalise back to the original scale of encoder inputs
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B: int, enc_out_list: List[torch.Tensor], x_list):
        # With channel_independence fixed to 0 in this pipeline, we only need
        # the shared multi-scale regression path, which removes unused branches
        # and slightly reduces overhead without changing behaviour.
        enc_inputs, out_res_list = x_list
        dec_out_list: List[torch.Tensor] = []
        for i, (enc_out, out_res) in enumerate(zip(enc_out_list, out_res_list)):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            dec_out = self.out_projection(dec_out, i, out_res)
            dec_out_list.append(dec_out)
        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise ValueError('Unsupported task name for TimeMixer')


# -----------------------------------------------------------------------------
# Pipeline configuration and training logic
# -----------------------------------------------------------------------------


@dataclass
class IronDailyConfig:
    # project_root: Path = Path(__file__).resolve().parents[0]
    # project_root: Path = Path(r"D:\清华工程博士\C3I\AutoMLAgent\openevolve\iron_test\exp_iron_4_gpu") 
    project_root: Path = Path(r"/home/jovyan/research/kaikai/c3i/AutoMLAgent/openevolve/iron_test/exp_iron_4_gpu") 
    checkpoint_dir: Path | None = None
    raw_data_override: str | None = None
    fusion_config: Dict[str, Any] | None = None
    cached_split_dir: Path | None = None
    use_cached_splits: bool = True
    seq_len: int = 48
    label_len: int = 0
    pred_len: int = 12
    freq: str = "b"
    target_col: str = "y"
    batch_size: int = 16
    learning_rate: float = 1e-2
    train_epochs: int = 10
    patience: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # device: str = "cpu"
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
    blend_alpha: float = 0.8

    def __post_init__(self) -> None:
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_root / "checkpoints"
        if self.fusion_config is None:
            self.fusion_config = copy.deepcopy(DEFAULT_FUSION_CONFIG)
        if self.cached_split_dir is None:
            self.cached_split_dir = self.project_root / "data"
        if self.split_ratio is None:
            self.split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cached_split_dir.mkdir(parents=True, exist_ok=True)

    @property
    def device_obj(self) -> torch.device:
        return torch.device(self.device)


def fuse_and_align_features(cfg: 'IronDailyConfig') -> pd.DataFrame:
    """Unused in this standalone pipeline; cached CSV splits are loaded instead."""
    raise NotImplementedError(
        "fuse_and_align_features is unused; cached train/val/test splits are loaded instead."
    )


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
    # unused helper; cached CSV splits are required
    raise NotImplementedError("compute_split_borders is disabled; cached train/val/test splits are required.")


def get_split_cache_paths(cfg: IronDailyConfig) -> Dict[str, Path]:
    names = ['train', 'val', 'test']
    return {name: cfg.cached_split_dir / f"{name}_raw.csv" for name in names}


def split_raw_dataframe(fused_df: pd.DataFrame, cfg: IronDailyConfig) -> Dict[str, pd.DataFrame]:
    # unused helper; cached train/val/test splits must be provided instead
    raise NotImplementedError("split_raw_dataframe is unused in this pipeline; cached splits must be provided.")


def load_splits_data(
    cfg: IronDailyConfig,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Path], bool]:
    split_paths = get_split_cache_paths(cfg)
    if cfg.use_cached_splits and all(path.exists() for path in split_paths.values()):
        logger.info("Loading cached splits from %s", cfg.cached_split_dir)
        splits = {
            name: pd.read_csv(path, parse_dates=['date']).sort_values('date').reset_index(drop=True)
            for name, path in split_paths.items()
        }
        return splits, split_paths


def run_feature_engineering_on_splits(
    raw_splits: Dict[str, pd.DataFrame], cfg: IronDailyConfig
) -> Dict[str, pd.DataFrame]:
    fe_splits: Dict[str, pd.DataFrame] = {}
    for name, df in raw_splits.items():
        fe_df = run_feature_engineering(df, cfg)
        fe_splits[name] = fe_df
    return fe_splits


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


def prepare_single_split_data(
    df: pd.DataFrame,
    cfg: IronDailyConfig,
    feature_cols: List[str] | None = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.assign(**{cfg.target_col: df.pop(cfg.target_col)})
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != 'date']
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected feature columns: {missing_cols}")
    df = df[['date'] + feature_cols]
    data_values = df[feature_cols].values.astype(np.float32)
    stamp_slice = build_time_mark_array(df['date'], cfg)
    split_entry = {
        'data': data_values,
        'stamp': stamp_slice.astype(np.float32),
        'length': len(data_values),
        'dates': df['date'].to_numpy(),
    }
    return split_entry, feature_cols


def prepare_splits_after_engineering(
    fe_splits: Dict[str, pd.DataFrame], cfg: IronDailyConfig
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    split_info: Dict[str, Dict[str, np.ndarray]] = {}
    feature_cols: List[str] | None = None
    for name in ['train', 'val', 'test']:
        if name not in fe_splits:
            raise KeyError(f"Missing split '{name}' in engineered datasets.")
        split_entry, feature_cols = prepare_single_split_data(fe_splits[name], cfg, feature_cols)
        split_info[name] = split_entry

    # 标准化除目标列之外的特征（使用训练集统计量），提高数值稳定性
    if 'train' in split_info:
        train_data = split_info['train']['data']
        if isinstance(train_data, np.ndarray) and train_data.ndim == 2 and train_data.shape[1] > 1:
            num_features = train_data.shape[1]
            feat_slice = slice(0, num_features - 1)  # 最后一列为目标y，保持原尺度
            mean = train_data[:, feat_slice].mean(axis=0, keepdims=True)
            std = train_data[:, feat_slice].std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            for name in ['train', 'val', 'test']:
                data = split_info[name]['data'].astype(np.float32)
                data[:, feat_slice] = (data[:, feat_slice] - mean) / std
                split_info[name]['data'] = data
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
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            worker_init_fn=_worker_init_fn,
            generator=_shared_generator,
        )
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


def _collect_log_forecasts(
    model: TimeMixer,
    loader: DataLoader,
    cfg: IronDailyConfig,
    device: torch.device,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Helper that returns (preds, trues, naive) in log space."""
    model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    naives: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)
            # 当前配置中总是使用多层下采样，因此解码器输入恒为 None
            dec_inp = None
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred_y, true_y = extract_target(outputs, batch_y, cfg)

            # Naive baseline: repeat last observed target value over the horizon
            if cfg.c_out == 1:
                last_val = batch_x[:, -1:, -1:]
            else:
                last_val = batch_x[:, -1:, 0:1]
            naive_y = last_val.repeat(1, cfg.pred_len, 1)

            preds.append(pred_y.cpu().numpy())
            trues.append(true_y.cpu().numpy())
            naives.append(naive_y.cpu().numpy())
    if not preds:
        return None, None, None

    preds_arr = np.concatenate(preds, axis=0)
    trues_arr = np.concatenate(trues, axis=0)
    naive_arr = np.concatenate(naives, axis=0)
    return preds_arr, trues_arr, naive_arr


def evaluate(
    model: TimeMixer,
    loader: DataLoader,
    cfg: IronDailyConfig,
    device: torch.device,
    apply_log_transform: bool = True,
    calibr: Tuple[float, float] | None = None,
) -> Tuple[float, float, float, float]:
    """Evaluate model on a loader and compute error metrics."""
    preds_arr, trues_arr, naive_arr = _collect_log_forecasts(model, loader, cfg, device)
    if preds_arr is None:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Blend model and naive forecasts in log space
    alpha = getattr(cfg, "blend_alpha", 0.8)
    preds_arr = alpha * preds_arr + (1.0 - alpha) * naive_arr

    # Optional linear calibration in log-space: y ≈ w * y_pred + b
    if calibr is not None:
        w, b = calibr
        preds_arr = preds_arr * float(w) + float(b)

    scaled_mse = np.mean((preds_arr - trues_arr) ** 2)
    scaled_mae = np.mean(np.abs(preds_arr - trues_arr))
    if apply_log_transform:
        pred_value = np.expm1(preds_arr)
        true_value = np.expm1(trues_arr)
    else:
        pred_value = preds_arr
        true_value = trues_arr
    value_mape = np.mean(
        np.abs((pred_value - true_value) / np.clip(true_value, 1e-6, None))
    )
    da_score = compute_directional_accuracy(pred_value, true_value)
    return scaled_mse, scaled_mae, value_mape, da_score


def compute_log_calibration(
    model: TimeMixer,
    loader: DataLoader,
    cfg: IronDailyConfig,
    device: torch.device,
) -> Tuple[float, float]:
    """Jointly tune blend_alpha and affine log-space calibration on validation data.

    We search a small grid of blend alphas; for each blended forecast we fit a
    simple linear calibration y ≈ w*y_pred + b in closed form, then keep the
    combination that minimises (MSE + MAE) on the validation set. To avoid
    harming performance, we also compare against an identity (no-calibration)
    mapping and only apply calibration when it helps.
    """
    preds_arr, trues_arr, naive_arr = _collect_log_forecasts(model, loader, cfg, device)
    if preds_arr is None:
        return 1.0, 0.0

    best_score = float("inf")
    best_alpha = float(getattr(cfg, "blend_alpha", 0.8))
    best_w, best_b = 1.0, 0.0

    # 扩大 alpha 搜索网格到 [0.0, 1.0] 并细化步长，提高在验证集上选择最优混合权重的精度
    candidate_alphas = [i / 20.0 for i in range(0, 21)]
    for alpha in candidate_alphas:
        blended = alpha * preds_arr + (1.0 - alpha) * naive_arr

        p = blended.reshape(-1)
        t = trues_arr.reshape(-1)
        mask = np.isfinite(p) & np.isfinite(t)
        if mask.sum() < 2:
            # 退化情况：无法稳定估计线性校准参数，回退到恒等映射
            w, b = 1.0, 0.0
        else:
            p_m = float(p[mask].mean())
            t_m = float(t[mask].mean())
            var_p = float(((p[mask] - p_m) ** 2).mean())
            if var_p <= 1e-12:
                # 几乎无方差时只校正偏移
                w = 1.0
                b = t_m - p_m
            else:
                cov_pt = float(((p[mask] - p_m) * (t[mask] - t_m)).mean())
                w = cov_pt / var_p
                b = t_m - w * p_m

        # 评价“学习得到的线性校准”和“无校准恒等映射”两种方案，避免校准在验证集上恶化误差
        calibrated = blended * float(w) + float(b)
        err = calibrated - trues_arr
        mse = float((err ** 2).mean())
        mae = float(np.abs(err).mean())
        score = mse + mae

        err_id = blended - trues_arr
        mse_id = float((err_id ** 2).mean())
        mae_id = float(np.abs(err_id).mean())
        score_id = mse_id + mae_id
        if score_id < score:
            score = score_id
            w, b = 1.0, 0.0

        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
            best_w, best_b = float(w), float(b)

    cfg.blend_alpha = best_alpha
    return best_w, best_b


def tune_blend_alpha(
    model: TimeMixer,
    loader: DataLoader,
    cfg: IronDailyConfig,
    device: torch.device,
) -> float:
    # Deprecated: blend_alpha is now tuned inside compute_log_calibration.
    return float(getattr(cfg, "blend_alpha", 0.8))


def train_predict_evaluate() -> None:
    cfg = IronDailyConfig()
    print("1) 加载训练集 验证集 测试集...")
    raw_splits, split_paths = load_splits_data(cfg)
    print(f"   已加载数据：{', '.join(str(p.name) for p in split_paths.values())}")

    print("   样本量：", {k: len(v) for k, v in raw_splits.items()})

    print("2) 特征工程：对拆分后的数据分别变换...")
    fe_splits = run_feature_engineering_on_splits(raw_splits, cfg)
    print("   特征工程完成，样本量：", {k: len(v) for k, v in fe_splits.items()})

    print("3) 数据窗口构建与标准化...")
    split_info, feature_cols = prepare_splits_after_engineering(fe_splits, cfg)
    enc_in = len(feature_cols)
    print(f"   输入特征维度 enc_in={enc_in}")
    loaders = make_dataloaders_from_splits(split_info, cfg)
    dataset_sizes = {name: len(loader.dataset) for name, loader in loaders.items()}
    print("   数据窗口数量：", dataset_sizes)

    print("4) 模型初始化与训练...")
    model = build_model(cfg, enc_in).to(cfg.device_obj)
    # 使用轻微的权重衰减提升泛化能力
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)
    criterion = nn.MSELoss()
    # 稍弱的方向正则，更好兼顾MSE/MAE与方向一致性
    lambda_dir = 0.08
    print(
        f"   训练参数：epochs={cfg.train_epochs}, lr={cfg.learning_rate}, "
        f"d_model={cfg.d_model}, d_ff={cfg.d_ff}, down_layers={cfg.down_sampling_layers}"
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
            # 简化：当前模型始终采用多尺度编码器，预测阶段不需要显式 decoder 输入
            dec_inp = None
            optimizer.zero_grad()
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred_y, true_y = extract_target(outputs, batch_y, cfg)
            mse_loss = criterion(pred_y, true_y)
            # 方向损失：鼓励预测价格变化方向与真实方向一致，以提高DA
            delta_pred = pred_y[:, 1:, :] - pred_y[:, :-1, :]
            delta_true = true_y[:, 1:, :] - true_y[:, :-1, :]
            dir_target = torch.sign(delta_true)
            dir_loss = F.relu(-delta_pred * dir_target).mean()
            loss = mse_loss + lambda_dir * dir_loss
            loss.backward()
            # 梯度裁剪以提高训练稳定性，避免偶发梯度爆炸影响预测精度
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    # 在验证集上联合搜索blend_alpha并拟合简单线性校准参数
    calib_w, calib_b = compute_log_calibration(model, loaders["val"], cfg, cfg.device_obj)

    print("5) 测试集评估...")
    test_mse, test_mae, test_mape, test_da = evaluate(
        model, loaders["test"], cfg, cfg.device_obj, calibr=(calib_w, calib_b)
    )
    print(
        f"   Test metrics -> scaled_MSE: {test_mse:.4f}, scaled_MAE: {test_mae:.4f}, "
        f"value_MAPE: {test_mape:.4f}, DA: {test_da:.4f}"
    )
# EVOLVE-BLOCK-END
    return test_mse, test_mae, test_mape, test_da

if __name__ == "__main__":
    test_mse, test_mae, test_mape, test_da = train_predict_evaluate()
    
