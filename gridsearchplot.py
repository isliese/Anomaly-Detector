import json
from functions import detect_anomalies_iqr, plot_anomaly_score
from gridsearchfunctions import load_csv, onechart_score
import time

# ✅ 저장된 best params 불러오기
def load_best_params(path="best_params_yfi.json"):
    with open(path, "r") as f:
        return json.load(f)

def run_iqr_and_plot(df, df_1chart, best_params):
    df_iqr = detect_anomalies_iqr(
        df.copy(),
        lookback=best_params["lookback"],
        k_ret=best_params["k_ret"],
        k_vol=best_params["k_vol"]
    )
    df_scored = onechart_score(df_iqr, df_1chart)

    threshold = df_scored["score"].quantile(0.95)
    mask = df_scored["score"] > threshold

    plot_anomaly_score(df_scored, idx=[100, 300], q=0.95)

    print("\n📊 Onechart Score 통계")
    print(f"전체 평균   : {df_scored['onechart_score'].mean():.4f}")
    print(f"이상치 평균 : {df_scored[mask]['onechart_score'].mean():.4f}")


if __name__ == "__main__":
    start_time = time.time()
    path_1min = "./Anomaly_Detection/candle_data/yfi.csv"       # 코인명 변경 필요
    path_1chart = "./Anomaly_Detection/candle_data/yfi_1chart.csv"      # 코인명 변경 필요

    df = load_csv(path_1min)
    df_1chart = load_csv(path_1chart)

    best_params = load_best_params()
    print("🎯 불러온 Best Params:", best_params)

    run_iqr_and_plot(df, df_1chart, best_params)

    end_time = time.time()
    print(f"\n⏱️ 총 실행 시간: {end_time - start_time:.2f}초")