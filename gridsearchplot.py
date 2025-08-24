import pandas as pd
import numpy as np
import time
from gridsearchfunctions import (
    load_csv,
    onechart_score,
    grid_search_anomaly,
    AnomalyScorer
)
from functions import detect_anomalies_iqr, plot_anomaly_score  # 재연님 코드

# AnomalyScorer 그리드서치 기반 best param 가져오기
def get_best_anomaly_scorer_params(df, df_1chart):
    param_grid = {
        "lookback_window": [10, 20, 30, 50],
        "threshold": [0.6, 0.7, 0.8],
        "price_weight": [0.3, 0.5, 0.7]
    }

    results = grid_search_anomaly(df, df_1chart, param_grid)
    best = results.iloc[0]

    return {
        "lookback": int(best["lookback_window"]),
        "k_ret": round(best["price_weight"], 2),
        "k_vol": round(best["volume_weight"], 2)
    }


# IQR 기반 이상치 탐지 및 시각화
def run_iqr_and_plot(df, df_1chart, best_params):
    df_iqr = detect_anomalies_iqr(
        df.copy(),
        lookback=best_params["lookback"],
        k_ret=best_params["k_ret"],
        k_vol=best_params["k_vol"]
    )
    df_scored = onechart_score(df_iqr, df_1chart)
    
    # 이상치 기준 마스크 생성
    threshold = df_scored["score"].quantile(0.95)
    mask = df_scored["score"] > threshold
    
    # 시각화
    plot_anomaly_score(df_scored, idx=[100, 300], q=0.95)
    
    # 결과 출력
    print("\n📊 Onechart Score 통계")
    print(f"전체 평균   : {df_scored['onechart_score'].mean():.4f}")
    print(f"이상치 평균 : {df_scored[mask]['onechart_score'].mean():.4f}")

# 실행 시작
if __name__ == "__main__":
    start_time = time.time()  # 시작 시간 기록
    path_1min = "./Anomaly_Detection/candle_data/YFI.csv"
    path_1chart = "./Anomaly_Detection/candle_data/YFI_1chart.csv"

    df = load_csv(path_1min)
    df_1chart = load_csv(path_1chart)

    best_params = get_best_anomaly_scorer_params(df, df_1chart)
    print("🎯AnomalyScorer 기반 Best Params (IQR 변환):", best_params)
    run_iqr_and_plot(df, df_1chart, best_params)
    
    end_time = time.time()  # 끝난 시간 기록
    print(f"\n⏱️ 총 실행 시간: {end_time - start_time:.2f}초")