import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import time
warnings.filterwarnings('ignore')

class EnhancedCryptoAnomalyDetector:
    def __init__(self, data, coin_name, lookback_window=100):
        self.data = data.copy()
        self.coin_name = coin_name
        self.lookback_window = lookback_window
        self.prepare_data()
        
        # 다양한 가중치 조합 정의
        self.weight_combinations = {
            'Balanced': {'price_change_abs': 0.4, 'volume_change_abs': 0.4, 'hl_spread': 0.2},
            'Price_Focus': {'price_change_abs': 0.6, 'volume_change_abs': 0.3, 'hl_spread': 0.1},
            'Volume_Focus': {'price_change_abs': 0.3, 'volume_change_abs': 0.6, 'hl_spread': 0.1},
            'Equal_Weight': {'price_change_abs': 0.33, 'volume_change_abs': 0.33, 'hl_spread': 0.34},
            'Spread_Focus': {'price_change_abs': 0.3, 'volume_change_abs': 0.3, 'hl_spread': 0.4}
        }
        
    def prepare_data(self):
        """데이터 전처리 및 특성 엔지니어링"""
        # timestamp를 datetime으로 변환
        self.data['datetime'] = pd.to_datetime(self.data['timestamp'], unit='ms', utc=True)
        
        # 기본 특성 계산
        self.data['price_change'] = self.data['close'].pct_change()
        self.data['price_change_abs'] = abs(self.data['price_change'])
        self.data['hl_spread'] = (self.data['high'] - self.data['low']) / self.data['close']
        self.data['volume_change'] = self.data['volume'].pct_change()
        self.data['volume_change_abs'] = abs(self.data['volume_change'])
        
    def rolling_anomaly_detection(self, features, weights, method='iqr', threshold=0.7):
        """롤링 윈도우 방식 이상치 탐지"""
        composite_scores = pd.Series(0.0, index=self.data.index)
        anomaly_flags = pd.Series(False, index=self.data.index)
        
        for i in range(self.lookback_window, len(self.data)):
            current_score = 0.0
            
            for feature in features:
                if feature not in self.data.columns:
                    continue
                
                # 과거 데이터만 사용
                historical_data = self.data[feature].iloc[i-self.lookback_window:i]
                current_value = self.data[feature].iloc[i]
                
                if pd.isna(current_value) or len(historical_data.dropna()) < 10:
                    continue
                
                # 이상 점수 계산
                if method == 'iqr':
                    Q1 = historical_data.quantile(0.25)
                    Q3 = historical_data.quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        upper_bound = Q3 + 1.5 * IQR
                        if current_value > upper_bound:
                            feature_score = min(1.0, (current_value - upper_bound) / IQR)
                        else:
                            feature_score = 0.0
                    else:
                        feature_score = 0.0
                        
                elif method == 'zscore':
                    hist_mean = historical_data.mean()
                    hist_std = historical_data.std()
                    if hist_std > 0:
                        z_score = abs(current_value - hist_mean) / hist_std
                        feature_score = min(1.0, z_score / 3.0)  # 0-1 스케일로 정규화
                    else:
                        feature_score = 0.0
                
                # 가중치 적용
                current_score += feature_score * weights.get(feature, 0.0)
            
            # 복합 점수 (0-1 스케일)
            composite_scores.iloc[i] = min(1.0, current_score)
            anomaly_flags.iloc[i] = current_score > threshold
        
        return composite_scores, anomaly_flags
    
    def analyze_all_combinations(self):
        """모든 가중치 조합에 대해 분석 수행"""
        results = {}
        
        for combo_name, weights in self.weight_combinations.items():
            # IQR 방식
            iqr_scores, iqr_flags = self.rolling_anomaly_detection(
                ['price_change_abs', 'volume_change_abs', 'hl_spread'], 
                weights, 
                method='iqr'
            )
            
            # Z-score 방식
            zscore_scores, zscore_flags = self.rolling_anomaly_detection(
                ['price_change_abs', 'volume_change_abs', 'hl_spread'], 
                weights, 
                method='zscore'
            )
            
            results[combo_name] = {
                'IQR': {
                    'scores': iqr_scores,
                    'flags': iqr_flags,
                    'anomaly_count': iqr_flags.sum(),
                    'avg_score': iqr_scores[iqr_scores > 0].mean() if (iqr_scores > 0).any() else 0,
                    'max_score': iqr_scores.max()
                },
                'Z-Score': {
                    'scores': zscore_scores,
                    'flags': zscore_flags,
                    'anomaly_count': zscore_flags.sum(),
                    'avg_score': zscore_scores[zscore_scores > 0].mean() if (zscore_scores > 0).any() else 0,
                    'max_score': zscore_scores.max()
                }
            }
        
        return results


def load_crypto_data_fast(file_path):
    """pandas를 사용한 빠른 데이터 로드"""
    try:
        df = pd.read_csv(file_path)
        # 필요한 컬럼만 선택
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_columns].sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"데이터 로드 오류 ({file_path}): {e}")
        return None


def main_analysis():
    """메인 분석 함수"""
    # 분석할 파일들
    files = {
        "candle_data/ETH.csv": "ETH",
        "candle_data/BERA.csv": "BERA", 
        "candle_data/YFI.csv": "YFI"
    }
    
    all_results = {}
    
    print("=" * 80)
    print("🔍 암호화폐 이상치 탐지 분석 결과")
    print("=" * 80)
    
    # 1. 각 코인별 분석 결과
    for file_path, coin_name in files.items():
        print(f"\n📊 {coin_name} 분석 결과")
        print("-" * 50)
        
        df = load_crypto_data_fast(file_path)
        if df is None:
            continue

        print(f"  데이터 크기: {df.shape[0]}행 x {df.shape[1]}열")
        
        print("\n         분석 중...         ")
            
        # 시간 측정
        start_time = time.time()
        detector = EnhancedCryptoAnomalyDetector(df, coin_name, lookback_window=100)
        coin_results = detector.analyze_all_combinations()
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"\n  ⏱ 분석 소요 시간: {elapsed:.2f}초")
        
        all_results[coin_name] = coin_results
        
        # 각 가중치 조합별 결과 출력
        for combo_name, combo_results in coin_results.items():
            print(f"\n  🎯 {combo_name} 가중치 결과:")
            for method, results in combo_results.items():
                print(f"     {method:8s} | 이상치: {results['anomaly_count']:3d}개 | 평균점수: {results['avg_score']:.3f} | 최고점수: {results['max_score']:.3f}")
    
    # 2. 가중치 조합별 코인 순위
    print(f"\n\n🏆 가중치 조합별 이상치 탐지 순위")
    print("=" * 80)
    
    for combo_name in detector.weight_combinations.keys():
        print(f"\n📈 {combo_name} 가중치 조합 순위")
        print("-" * 50)
        
        # IQR 방식 순위
        iqr_ranking = []
        zscore_ranking = []
        
        for coin_name in files.values():
            if coin_name in all_results:
                iqr_count = all_results[coin_name][combo_name]['IQR']['anomaly_count']
                zscore_count = all_results[coin_name][combo_name]['Z-Score']['anomaly_count']
                
                iqr_ranking.append((coin_name, iqr_count))
                zscore_ranking.append((coin_name, zscore_count))
        
        # 순위 정렬
        iqr_ranking.sort(key=lambda x: x[1], reverse=True)
        zscore_ranking.sort(key=lambda x: x[1], reverse=True)
        
        print("  IQR 방식:")
        for rank, (coin, count) in enumerate(iqr_ranking, 1):
            print(f"    {rank}위: {coin:4s} - {count:3d}개")
        
        print("  Z-Score 방식:")
        for rank, (coin, count) in enumerate(zscore_ranking, 1):
            print(f"    {rank}위: {coin:4s} - {count:3d}개")
    
    # 3. 요약 통계
    print(f"\n\n📋 전체 요약 통계")
    print("=" * 80)
    
    for coin_name in files.values():
        if coin_name not in all_results:
            continue
            
        print(f"\n{coin_name}:")
        total_iqr = sum(all_results[coin_name][combo]['IQR']['anomaly_count'] 
                       for combo in detector.weight_combinations.keys())
        total_zscore = sum(all_results[coin_name][combo]['Z-Score']['anomaly_count'] 
                          for combo in detector.weight_combinations.keys())
        
        print(f"  전체 IQR 이상치: {total_iqr}개")
        print(f"  전체 Z-Score 이상치: {total_zscore}개")
        
        # 가장 효과적인 조합 찾기
        best_iqr_combo = max(detector.weight_combinations.keys(), 
                            key=lambda x: all_results[coin_name][x]['IQR']['anomaly_count'])
        best_zscore_combo = max(detector.weight_combinations.keys(),
                               key=lambda x: all_results[coin_name][x]['Z-Score']['anomaly_count'])
        
        print(f"  최적 IQR 조합: {best_iqr_combo} ({all_results[coin_name][best_iqr_combo]['IQR']['anomaly_count']}개)")
        print(f"  최적 Z-Score 조합: {best_zscore_combo} ({all_results[coin_name][best_zscore_combo]['Z-Score']['anomaly_count']}개)")


if __name__ == "__main__":
    try:
        main_analysis()
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        print("파일 경로와 데이터 형식을 확인해주세요.")