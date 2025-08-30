import pandas as pd
import numpy as np
import mplfinance as mpf
from itertools import product
from scipy.stats import spearmanr

# ---------------------------
# 1. CSV 로드 함수
# ---------------------------
def load_csv(path, timestamp_col="timestamp"):
    df = pd.read_csv(path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms', utc=True)
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df.set_index(timestamp_col, inplace=True)
    return df

# ---------------------------
# 2. 변동성 → 파라미터 매핑 함수 (rule-based)
# ---------------------------
def volatility_to_params(df_1min):
    vol = df_1min['close'].pct_change().std()
    lookback = int(np.clip(20 + vol*1000, 10, 100)) #  변동성이 크면 lookback 길게
    threshold = np.clip(0.7 + vol*2, 0.6, 0.95) # 변동성 높으면 threshold도 높게
    return lookback, threshold

# ---------------------------
# 3. 1차트 변동성 점수
# ---------------------------
def onechart_score(df_1min, df_1chart, shift=1, lookback=100, k=1.5):
    df_score = df_1min.copy()
    per_min_std = df_1chart['fx_diff'].astype(float).resample("1min").std().reindex(df_score.index)
    base = per_min_std.shift(shift)
    Q1 = base.rolling(lookback, min_periods=1).quantile(0.25)
    Q3 = base.rolling(lookback, min_periods=1).quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + k*IQR
    over = (per_min_std - upper).clip(lower=0)
    df_score["onechart_volatility"] = over / (IQR + 1e-12)
    df_score["onechart_volatility"] = df_score["onechart_volatility"].fillna(0)
    return df_score

# ---------------------------
# 4. IQR + Z-score + 1차트 통합 이상치 점수
# ---------------------------
def detect_anomalies_combined(df_1min, df_1chart, lookback=100, k_ret=1.5):
    df = df_1min.copy()
    df["ret"] = df["close"].pct_change().fillna(0)

    iqr_scores = []
    z_scores = []

    for i in range(len(df)):
        start = max(0, i-lookback)
        window = df["ret"].iloc[start:i+1]

        # IQR score
        q1 = window.quantile(0.25)
        q3 = window.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k_ret*iqr
        upper = q3 + k_ret*iqr
        val = df["ret"].iloc[i]
        iqr_score = max(0, val - upper) + max(0, lower - val)
        iqr_scores.append(iqr_score)

        # Z-score
        hist_mean = window.mean()
        hist_std = window.std()
        if hist_std > 0:
            z_score = abs(val - hist_mean) / hist_std
            z_score = min(1.0, z_score / 3.0)  # 0~1로 정규화
        else:
            z_score = 0.0
        z_scores.append(z_score)

    df["iqr_score"] = iqr_scores
    df["z_score"] = z_scores

    # 1차트 변동성 점수
    df_score = onechart_score(df, df_1chart)

    # 최종 가중합 (2가지 버전)
    df_score["final_conf_0.5_0.3_0.2"] = (
        df_score["iqr_score"]*0.5 +
        df_score["onechart_volatility"]*0.3 +
        df_score["z_score"]*0.2
    )
    df_score["final_conf_0.4_0.4_0.2"] = (
        df_score["iqr_score"]*0.4 +
        df_score["onechart_volatility"]*0.4 +
        df_score["z_score"]*0.2
    )

    return df_score

# ---------------------------
# 5. Grid Search 
# ---------------------------
def grid_search_combined(df_1min, df_1chart, lookback_list, threshold_list, weight_list):
    results = []
    for lookback, threshold, price_weight in product(lookback_list, threshold_list, weight_list):
        df_score = detect_anomalies_combined(df_1min, df_1chart, lookback=lookback, k_ret=threshold)
        a = df_score["iqr_score"].fillna(0)*price_weight + df_score["z_score"].fillna(0)*(1-price_weight)
        b = df_score["onechart_volatility"].fillna(0)
        corr, _ = spearmanr(a, b)
        results.append({
            "lookback": lookback,
            "threshold": threshold,
            "price_weight": price_weight,
            "volume_weight": 1-price_weight,
            "correlation": corr
        })
    df_result = pd.DataFrame(results).sort_values("correlation", ascending=False)
    return df_result

# ---------------------------
# 6. 시각화
# ---------------------------
def plot_anomaly_score(df, idx, q=0.95, anomaly_key="final_conf_0.5_0.3_0.2", title=""):
    threshold = df[anomaly_key].quantile(q)
    score_points = np.where(df[anomaly_key] > threshold, df['close'], np.nan)
    add_plots = [mpf.make_addplot(score_points[idx[0]:idx[1]], type='scatter', marker='o', markersize=10, color='blue')]
    chart_title = title if title else f"Anomaly Detection (Threshold: {threshold:.4f})"
    mpf.plot(df.iloc[idx[0]:idx[1]], type='candle', style='yahoo', addplot=add_plots, volume=True, title=chart_title)

# ---------------------------
# 7. 실행 
# ---------------------------
if __name__ == "__main__":
    paths_1min = {'./candle_data/ETH.csv': "ETH", './candle_data/BERA.csv': "BERA"}
    for path_1min, symbol in paths_1min.items():
        print("------------------------------------")
        print(f"\n0. {symbol} 데이터 가져오는 중...")
        path_1chart = path_1min.replace(".csv","_1chart.csv")
        df_1min = load_csv(path_1min)
        df_1chart = load_csv(path_1chart)

        # 변동성 기반 파라미터
        print("1. 변동성 기반 파라미터 조정 중...")
        lookback, threshold = volatility_to_params(df_1min)

        # GridSearch 범위
        print("2. GridSearch 하는 중...")
        lookback_list = [lookback-10, lookback, lookback+10]
        threshold_list = [threshold-0.05, threshold, threshold+0.05]
        weight_list = [0.3, 0.5, 0.7]

        search_result = grid_search_combined(df_1min, df_1chart, lookback_list, threshold_list, weight_list)
        print(f"\n📌 [{symbol}] GridSearch Top 5 Results:")
        print(search_result.head())
        
        best_lookback = int(search_result.iloc[0]["lookback"])
        print(f"Grid Search 결과 {symbol} 종목은 lookback이 {best_lookback}일 때 제일 안정적이라고 판단")  

        # 최적 파라미터로 이상치 탐지
        print("\n3. 최적 파라미터로 이상치 탐지하는 중...")
        best = search_result.iloc[0]
        df_final = detect_anomalies_combined(df_1min, df_1chart, lookback=int(best["lookback"]), k_ret=best["threshold"])

        # 결과 시각화
        print("4. 결과 시각화하는 중...\n\n\n")
        plot_anomaly_score(df_final, idx=[100,300], q=0.95, anomaly_key="final_conf_0.5_0.3_0.2",
                           title=f"{symbol} - Final Anomaly Score (0.5+0.3+0.2)")
