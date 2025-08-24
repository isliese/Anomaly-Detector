# anomaly_detection_pipeline.py

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr
from itertools import product


# ============================
# 📌 1. 데이터 로딩 함수
# ============================

def load_csv(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    return df


# ============================
# 📌 2. 틱 기반 스코어 계산
# ============================

def onechart_score(min_df, onechart_df, std_count_scaling=False, shift=1, lookback=100, k=1.5):
    out = min_df.copy()

    fx_diff = onechart_df["fx_diff"].astype(float)

    per_min_std = fx_diff.resample("1min", label="left", closed="left").std().reindex(out.index)
    per_min_tick_count = fx_diff.resample("1min", label="left", closed="left").count().reindex(out.index)

    if std_count_scaling:
        per_min_std = per_min_std / np.sqrt(per_min_tick_count.astype(float))

    base = per_min_std.shift(shift)
    Q1 = base.rolling(lookback).quantile(0.25)
    Q3 = base.rolling(lookback).quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + k * IQR

    over = (per_min_std - upper).clip(lower=0)
    score = over / (IQR.abs() + 1e-12)

    out["1min_std"] = per_min_std
    out["1min_tick_count"] = per_min_tick_count
    out["onechart_score"] = score

    return out


# ============================
# 📌 3. AnomalyScorer 클래스
# ============================

class AnomalyScorer:
    def __init__(self, df, coin_name="UNKNOWN"):
        self.df = df.copy()
        self.coin_name = coin_name
        self._prepare_data()

    def _prepare_data(self):
        self.df['price_change'] = self.df['close'].pct_change()
        self.df['price_change_abs'] = self.df['price_change'].abs()
        self.df['volume_change'] = self.df['volume'].pct_change()
        self.df['volume_change_abs'] = self.df['volume_change'].abs()
        self.df['hl_spread'] = (self.df['high'] - self.df['low']) / self.df['close']
        self.df.dropna(inplace=True)

    def compute_realtime_score(self, lookback_window=30, threshold=0.7,
                               price_weight=0.6, volume_weight=0.4):
        scores = []
        flags = []

        for i in range(lookback_window, len(self.df)):
            hist = self.df.iloc[i - lookback_window:i]
            current = self.df.iloc[i]

            price_mean = hist['price_change_abs'].mean()
            price_std = hist['price_change_abs'].std()
            volume_mean = hist['volume'].mean()
            volume_std = hist['volume'].std()

            price_score = abs(current['price_change_abs'] - price_mean) / (price_std + 1e-8)
            volume_score = abs(current['volume'] - volume_mean) / (volume_std + 1e-8)

            composite_score = min(1.0, (price_score * price_weight + volume_score * volume_weight) / 5)
            is_anomaly = composite_score > threshold

            scores.append(composite_score)
            flags.append(is_anomaly)

        self.df.loc[self.df.index[lookback_window:], 'realtime_score'] = scores
        self.df.loc[self.df.index[lookback_window:], 'realtime_flag'] = pd.Series(flags, dtype=bool).values

        return self.df[['realtime_score', 'realtime_flag']].dropna()


# ============================
# 📌 4. 평가 함수 (틱 기반과 상관관계)
# ============================

def evaluate_scorer(anomaly_df, onechart_df):
    scored_df = onechart_score(anomaly_df, onechart_df)
    a = scored_df["realtime_score"].fillna(0)
    b = scored_df["onechart_score"].fillna(0)
    corr, _ = spearmanr(a, b)
    return corr


# ============================
# 📌 5. 그리드 서치로 파라미터 튜닝
# ============================

def grid_search_anomaly(df, onechart_df, param_grid):
    results = []

    for win, thresh, price_w in product(
        param_grid['lookback_window'],
        param_grid['threshold'],
        param_grid['price_weight']
    ):
        volume_w = 1 - price_w

        scorer = AnomalyScorer(df.copy())
        try:
            _ = scorer.compute_realtime_score(
                lookback_window=win,
                threshold=thresh,
                price_weight=price_w,
                volume_weight=volume_w
            )
            corr = evaluate_scorer(scorer.df, onechart_df)

            results.append({
                "lookback_window": win,
                "threshold": thresh,
                "price_weight": price_w,
                "volume_weight": volume_w,
                "correlation": corr
            })
        except Exception as e:
            print(f"Error at win={win}, thresh={thresh}, weight={price_w}: {e}")
            continue

    return pd.DataFrame(results).sort_values("correlation", ascending=False)


# ============================
# ✅ 6. 실행 (main block)
# ============================

if __name__ == "__main__":
    path_1min = "./Anomaly_Detection/candle_data/YFI.csv"
    path_1chart = "./Anomaly_Detection/candle_data/YFI_1chart.csv"

    df_1min = load_csv(path_1min)
    df_1chart = load_csv(path_1chart)

    param_grid = {
        "lookback_window": [10, 20, 30, 50],
        "threshold": [0.6, 0.7, 0.8],
        "price_weight": [0.3, 0.5, 0.7]
    }

    search_result = grid_search_anomaly(df_1min, df_1chart, param_grid)

    print("\n📌 Best Parameters (based on onechart correlation):")
    print(search_result.head(5))