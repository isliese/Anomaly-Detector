from functions import *
import matplotlib

matplotlib.use('TkAgg',force=True)
print("Switched to:",matplotlib.get_backend())

# 1분봉 캔들 차트 
paths_1min = {'./candle_data/ETH.csv': "ETH", 
              './candle_data/BERA.csv': "BERA"}

 # 거래 tic별 가격 변화량 
paths_1chart = {'./candle_data/ETH_1chart.csv': "ETH_1chart",
                './candle_data/BERA_1chart.csv': "BERA_1chart"}

for path_1min in paths_1min.keys():
    symbol = paths_1min[path_1min]
    
    df = load_csv(path_1min)
    path_1chart = path_1min[0:-4]+"_1chart.csv"
    df_1chart = load_csv(path_1chart)

    # ret = 수익률 score = 이상치 점수, is_anomaly = 최종 이상 여부
    print("")
    df_iqr = detect_anomalies_iqr(df.copy(), lookback=100) # 최근 100분 데이터를 기준으로 Q1, Q3, IQR 계산
    threshold = df_iqr["score"].quantile(0.95) # 상위 5% 기준으로 threshold 설정
    mask = df_iqr["score"] > threshold
    
    # 종목명을 포함한 제목으로 그래프 표시
    chart_title = f"{symbol} - Anomaly Detection (Top 5%)"
    plot_anomaly_score(df_iqr, idx=[100, 300], q=0.95, title=chart_title)

    # 변동성(표준편차) 기반 점수 => 특정 일 분 간 tick 데이터 변동성이 평소보다 얼마나 튀었는지
    df_score = onechart_score(df_iqr, df_1chart)

    # 결과 출력 
    print(f"[{symbol}] 일분봉차트 이상치 점수 평균:", df_score["onechart_score"].mean())
    print(f"[{symbol}] IQR 이상치 점수가 높은 구간에서만의 일분봉차트 이상치 점수 평균:", df_score[mask]["onechart_score"].mean())