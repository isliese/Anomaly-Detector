import pandas as pd
import numpy as np
import mplfinance as mpf
from itertools import product
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import time

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
# 2. 1차트 변동성 점수(IQR)
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
# 3. Z-score + 1차트 통합 이상치 점수
# ---------------------------
def detect_anomalies_combined(df_1min, df_1chart, lookback=100, k_ret=1.5):
    df = df_1min.copy()
    df["ret"] = df["close"].pct_change().fillna(0)

    z_scores = []

    for i in range(len(df)):
        start = max(0, i - lookback)
        # 만약 i가 lookback보다 작은 경우에는 자료가 누적될 때까지 기다림
        if i < lookback:
            window = df["ret"].iloc[:i+1]
        else:
            window = df["ret"].iloc[i - lookback + 1:i + 1]  # lookback 이후에는 정상적으로 참조

        val = df["ret"].iloc[i]

        # Z-score 계산
        hist_mean = window.mean()
        hist_std = window.std()
        if hist_std > 0:
            z_score = abs(val - hist_mean) / hist_std
            z_score = min(1.0, z_score / 3.0)  # 0~1로 정규화
        else:
            z_score = 0.0
        z_scores.append(z_score)

    df["z_score"] = z_scores

    # 1차트 변동성 점수
    df_score = onechart_score(df, df_1chart)

    return df_score

# ---------------------------
# 4. Grid Search 
# ---------------------------
def grid_search_combined(df_1min, df_1chart, lookback_list, threshold_list, weight_list):
    results = []
    for lookback, threshold, price_weight in product(lookback_list, threshold_list, weight_list):
        df_score = detect_anomalies_combined(df_1min, df_1chart, lookback=lookback, k_ret=threshold)
        a = df_score["z_score"].fillna(0)
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
# 5. 시각화
# ---------------------------
def plot_anomaly_score(df, idx, q=0.95, anomaly_key="z_score", title=""):
    threshold = df[anomaly_key].quantile(q)
    score_points = np.where(df[anomaly_key] > threshold, df['close'], np.nan)
    add_plots = [mpf.make_addplot(score_points[idx[0]:idx[1]], type='scatter', marker='o', markersize=10, color='blue')]
    chart_title = title if title else f"Anomaly     Detection (Threshold: {threshold:.4f})"
    mpf.plot(df.iloc[idx[0]:idx[1]], type='candle', style='yahoo', addplot=add_plots, volume=True, title=chart_title)

# ---------------------------
# 6. 상관계수 Heatmap 그리기
# ---------------------------
def plot_correlation_heatmap(results_df):
    # 'lookback'을 세로축으로, 'price_weight'을 가로축으로, 'correlation' 값을 표시
    pivot_table = results_df.pivot_table(
        index="lookback",  # 세로축: lookback
        columns="price_weight",   # 가로축: price_weight
        values="correlation",     # 값: correlation
        aggfunc="mean"            # 평균값으로 집계
    )

    # Heatmap 시각화
    plt.figure(figsize=(8, 6))  # 크기 설정

    # 히트맵을 그리면서 스타일 조정
    sns.heatmap(pivot_table, 
                annot=True, 
                fmt=".6f",  # 소수점 3자리로 포맷팅
                cmap="YlGnBu",  # 색상 팔레트
                cbar_kws={'label': 'Spearman Correlation'},  # 색상 막대 레이블
                linewidths=0.8,  # 격자선 두께
                square=True)  # 정사각형 비율 유지

    # 제목, 축 레이블 설정
    plt.title(f"{coin} Grid Search Correlation Heatmap", fontsize=14, weight='bold')
    plt.xlabel("Price Weight", fontsize=12, labelpad=10)
    plt.ylabel("Lookback Window", fontsize=12, labelpad=10)

    # 여백 조정
    plt.tight_layout()
    plt.show()

# ---------------------------
# 7. 실행 
# ---------------------------
if __name__ == "__main__":
    start_time = time.time()
    coin = 'yfi'  # 간편하게 변경할 수 있도록 변수 설정
    
    paths_1min = {f'./candle_data/{coin}.csv': f"{coin}"}
    for path_1min, symbol in paths_1min.items():
        print("------------------------------------")
        print(f"\n1. {symbol} 데이터 가져오는 중...")
        path_1chart = path_1min.replace(".csv","_1chart.csv")
        df_1min = load_csv(path_1min)
        df_1chart = load_csv(path_1chart)

        # GridSearch 범위
        print("2. GridSearch 하는 중...")
        lookback_list = [50, 70, 100]
        threshold_list = [0.7]
        weight_list = [0.3, 0.5, 0.7]

        # Grid Search 수행
        search_result = grid_search_combined(df_1min, df_1chart, lookback_list, threshold_list, weight_list)
        print(f"\n📌 [{symbol}] GridSearch Top 5 Results:")
        print(search_result.head())

        # GridSearch 결과에서 가장 높은 상관계수를 가진 파라미터 선택
        search_result = search_result.sort_values("correlation", ascending=False)
        best_result = search_result.iloc[0]

        # 최적 파라미터
        best_lookback = int(best_result["lookback"])
        best_threshold = best_result["threshold"]
        best_price_weight = best_result["price_weight"]

        # 최적 price_weight에 맞는 anomaly_key 선택
        # 여기서는 anomaly_key를 단순히 "z_score"로 변경했음
        best_anomaly_key = "z_score"  # 가중합 없이 z_score만 사용

        print(f"Grid Search 결과 {symbol} 종목은 lookback이 {best_lookback}일 때 제일 안정적이라고 판단")  

        # 최적 파라미터로 이상치 탐지
        print("\n3. 최적 파라미터로 이상치 탐지하는 중...")
        df_final = detect_anomalies_combined(df_1min, df_1chart, lookback=best_lookback, k_ret=best_threshold)

        # 결과 시각화
        print("4. 결과 시각화하는 중...")
        plot_anomaly_score(df_final, idx=[100, 300], q=0.95, anomaly_key=best_anomaly_key,
                           title=f"{symbol} - Final Anomaly Score")
        
        # 상관계수 Heatmap 그리기
        print("5. 상관계수 Heatmap 시각화 중...\n\n\n")
        plot_correlation_heatmap(search_result)  # 최적의 파라미터에 대한 결과로 heatmap 그리기
        
        end_time = time.time()
        print(f"\n⏱️ 총 실행 시간: {end_time - start_time:.2f}초")