import json
from functions import onechart_score, detect_anomalies_iqr, plot_anomaly_score  # onechart_score는 min_periods=lookback의 차이였기 때문에 재연님 방식으로 함수 삽입(더 엄밀함)
from gridsearchfunctions import load_csv
import time

# ✅ 저장된 best params 불러오기

coin  = "YFI"       ## 코인명 변경 필요!!

def load_best_params(path=f"best_params_{coin}.json"):
    with open(path, "r") as f:
        return json.load(f)

def run_iqr_and_plot(df, df_1chart, best_params):       # ↓ 재연님 코드와 동일
    df_iqr = detect_anomalies_iqr(
        df.copy(),
        lookback=best_params["lookback"],               # ↓ 파라미터 업데이트(종완)
        k_ret=best_params["k_ret"],                     # (이 부분만 다름 : 파라미터 조정)
        k_vol=best_params["k_vol"]                      # ↑ 파라미터 업데이트(종완)
    )
    df_scored = onechart_score(df_iqr, df_1chart)       # 이상치 score

    threshold = df_scored["score"].quantile(0.95)
    mask = df_scored["score"] > threshold

    plot_anomaly_score(df_scored, idx=[100, 300], q=0.95)
    
    print("\n📊 Onechart Score 통계")
    print(f"전체 평균   : {df_scored['onechart_score'].mean():.4f}")
    print(f"이상치 평균 : {df_scored[mask]['onechart_score'].mean():.4f}")   # ↑ 여기까지 재연님 코드와 동일(이상치 점수화)


if __name__ == "__main__":
    start_time = time.time()
    path_1min = f"./Anomaly_Detection/candle_data/{coin}.csv"
    path_1chart = f"./Anomaly_Detection/candle_data/{coin}_1chart.csv"   
    
    df = load_csv(path_1min)
    df_1chart = load_csv(path_1chart)

    best_params = load_best_params()
    print("🎯 불러온 Best Params:", best_params)
    
    run_iqr_and_plot(df, df_1chart, best_params)    # 재연님 코드와 동일 (시각화)

    end_time = time.time()
    print(f"\n⏱️ 총 실행 시간: {end_time - start_time:.2f}초")