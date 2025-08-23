from functions_CR import *
import pandas as pd

# 일단은 BERA, ETH, YFI만 실행
file_path_dic = {
    # r"candle_data\ADA.csv": "ADA", 
    r"candle_data\BERA.csv": "BERA", 
    r"candle_data\ETH.csv": "ETH", 
    # r"candle_data\PENGU.csv": "PENGU",
    # r"candle_data\SOON.csv": "SOON", 
    # r"candle_data\SUNDOG.csv": "SUNDOG", 
    # r"candle_data\XRP.csv": "XRP", 
    r"candle_data\YFI.csv": "YFI"
}


def basic_usage():
    """기본적인 사용 방법"""
    print("=== 기본 사용법 예제 ===")
    
    # 데이터 로드
    for file_path in file_path_dic.keys():
        coin_name = file_path_dic[file_path]
        
        df = load_crypto_data(file_path)
        if df is not None:
            # 탐지기 초기화
            detector = CryptoAnomalyDetector(df, coin_name, lookback_window=100)
            
            # 이상치 탐지 실행
            result1 = detector.method1_statistical_outliers_rolling()
            print(f"통계적 이상치: {result1['total_outliers']}개")
            
            # 복합 점수 계산
            scores, threshold, count = detector.method4_composite_anomaly_score_rolling()
            print(f"복합 이상치: {count}개")
            
            # 실시간 모델 실행
            anomaly_scores, flags, total = detector.real_time_anomaly_model()
            print(f"실시간 이상치: {total}개")
            
            # 요약 정보 출력
            summary = detector.get_anomaly_summary()
            print(f"요약: {summary}")

# 빠른 분석
def quick_analysis():
    """빠른 분석 예제"""
    print("\n=== 빠른 분석 예제 ===")
    
    for file_path in file_path_dic.keys():
        coin_name = file_path_dic.items()
    
        detector, summary = quick_anomaly_detection(file_path, coin_name)
        if detector:
            print(f"{coin_name} 분석 완료:")
            print(f"- 총 이상치: {summary.get('total_anomalies', 0)}개")
            print(f"- 이상치 비율: {summary.get('anomaly_rate', 0):.2f}%")
            print(f"- 최근 점수: {summary.get('latest_score', 0):.4f}")

# 여러 코인 일괄 분석
def batch_analysis():
    """여러 코인 일괄 분석 예제"""
    print("\n=== 일괄 분석 예제 ===")
    
    # 분석할 코인들
    coin_files = {
        "candle_data/ETH.csv": "ETH",
        "candle_data/BERA.csv": "BERA",
        "candle_data/YFI.csv": "YFI"
    }
    
    results = {}
    summaries = {}
    
    for file_path, coin_name in coin_files.items():
        try:
            detector, summary = quick_anomaly_detection(file_path, coin_name)
            if detector:
                results[coin_name] = summary.get('total_anomalies', 0)
                summaries[coin_name] = summary
                print(f"✅ {coin_name}: {summary.get('total_anomalies', 0)}개 이상치")
        except Exception as e:
            print(f"❌ {coin_name} 분석 실패: {e}")
            results[coin_name] = 0
    
    # 순위 출력
    method_results = {"실시간 이상치 탐지": results}
    print_ranking_summary(method_results)

# 실시간 모니터링 시뮬레이션
def realtime_monitoring():
    """실시간 모니터링 시뮬레이션"""
    print("\n=== 실시간 모니터링 시뮬레이션 ===")
    
    for file_path in file_path_dic.keys():
        coin_name = file_path_dic.items()
    
        df = load_crypto_data(file_path)
        if df is not None:
            detector = CryptoAnomalyDetector(df, coin_name, lookback_window=50)
            
            # 실시간 모델 실행
            detector.real_time_anomaly_model(lookback_window=30)
            
            # 최근 10개 시점의 점수 확인
            recent_scores = detector.data['realtime_anomaly_score'].tail(10)
            recent_flags = detector.data['realtime_anomaly_flag'].tail(10)
            recent_times = detector.data['datetime'].tail(10)
            
            print(f"\n📊 {coin_name} 최근 10분간 이상치 점수:")
            for i, (time, score, flag) in enumerate(zip(recent_times, recent_scores, recent_flags)):
                status = "🚨 이상치!" if flag else "✅ 정상"
                print(f"   {time.strftime('%H:%M:%S')} - 점수: {score:.4f} {status}")
            
            # 가장 최근 점수 확인
            latest_score = detector.get_latest_anomaly_score()
            print(f"\n🔍 현재 이상치 점수: {latest_score:.4f}")
            if latest_score > 0.7:
                print("⚠️ 높은 이상치 점수 감지!")

# 특정 컬럼만 분석
def custom_analysis():
    """커스터마이징된 분석 예제"""
    print("\n=== 커스터마이징 분석 예제 ===")
    
    for file_path in file_path_dic.keys():
        coin_name = file_path_dic.items()
    
        df = load_crypto_data(file_path)
        if df is not None:
            detector = CryptoAnomalyDetector(df, coin_name, lookback_window=200)
            
            # 가격 변동만 분석
            result_price = detector.method1_statistical_outliers_rolling(
                columns=['price_change_abs'], 
                threshold_method='iqr', 
                iqr_multiplier=2.0
            )
            
            # 거래량 변동만 분석  
            result_volume = detector.method1_statistical_outliers_rolling(
                columns=['volume_change_abs'],
                threshold_method='zscore',
                z_threshold=2.5
            )
            
            print(f"가격 변동 이상치: {result_price['price_change_abs']['count']}개")
            print(f"거래량 변동 이상치: {result_volume['volume_change_abs']['count']}개")
            
            # 결과 데이터프레임 저장
            detector.data.to_csv(f"{coin_name}_anomaly_results.csv", index=False)
            print(f"결과가 {coin_name}_anomaly_results.csv에 저장되었습니다.")

# 데이터 필터링 후 분석
def filtered_analysis():
    """특정 기간 데이터만 분석하는 예제"""
    print("\n=== 필터링된 분석 예제 ===")
    
    for file_path in file_path_dic.keys():
        coin_name = file_path_dic.items()
        
        df = load_crypto_data(file_path)
        if df is not None:
            # 타임스탬프를 datetime으로 변환
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # 최근 1000개 데이터만 사용
            df_recent = df.tail(1000).reset_index(drop=True)
            
            detector = CryptoAnomalyDetector(df_recent, f"{coin_name}_Recent", lookback_window=50)
            
            # 빠른 분석
            result = detector.method1_statistical_outliers_rolling()
            summary = detector.get_anomaly_summary()
            
            print(f"최근 데이터 분석 결과:")
            print(f"- 분석 기간: {summary['analysis_periods']}분")
            print(f"- 총 이상치: {result['total_outliers']}개")


# 메인 실행 함수
if __name__ == "__main__":
    try:
        basic_usage()
        quick_analysis() 
        batch_analysis()
        realtime_monitoring()
        custom_analysis()
        filtered_analysis()
        
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        print("파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"실행 중 오류가 발생했습니다: {e}")