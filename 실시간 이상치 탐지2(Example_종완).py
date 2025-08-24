import pandas as pd
import numpy as np
from functions import load_csv, plot_anomaly_score, onechart_score, detect_anomalies_iqr

# 🔍 Grid Search 최적화 함수
def grid_search_best_params(df, df_1chart, param_grid):
    best_lift = -np.inf
    best_params = {}
    best_result = None

    for lookback in param_grid["lookback"]:
        for k_ret in param_grid["k_ret"]:
            for k_vol in param_grid["k_vol"]:
                df_iqr = detect_anomalies_iqr(
                    df.copy(),
                    lookback=lookback,
                    k_ret=k_ret,
                    k_vol=k_vol
                )
                df_scored = onechart_score(df_iqr, df_1chart)

                if df_scored["onechart_score"].isnull().all():
                    continue

                threshold = df_scored["score"].quantile(0.95)
                mask = df_scored["score"] > threshold

                mean_all = df_scored["onechart_score"].mean()
                mean_anomaly = df_scored[mask]["onechart_score"].mean()

                if mean_all > 0:
                    lift = mean_anomaly / mean_all
                    if lift > best_lift:
                        best_lift = lift
                        best_params = {
                            "lookback": lookback,
                            "k_ret": k_ret,
                            "k_vol": k_vol
                        }
                        best_result = df_scored

    return best_params, best_lift, best_result


# 🚀 실행 예시
if __name__ == "__main__":
    path_1min = './Anomaly_Detection/candle_data/YFI.csv'
    path_1chart = './Anomaly_Detection/candle_data/YFI_1chart.csv'

    df = load_csv(path_1min)
    df_1chart = load_csv(path_1chart)

    param_grid = {
        "lookback": [10, 20, 30, 50],
        "k_ret": [0.6, 0.7, 0.8],
        "k_vol": [0.3, 0.5, 0.7]
    }

    best_params, best_lift, best_result = grid_search_best_params(df, df_1chart, param_grid)

    print("✅ 최적 파라미터:", best_params)
    print(f"📈 Onechart Lift: {best_lift:.2f}")

    # 시각화
    plot_anomaly_score(best_result, idx=[100, 300], q=0.95)