# crypto_anomaly_detector.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

class CryptoAnomalyDetector:
    def __init__(self, data, coin_name, lookback_window=100):
        self.data = data.copy()
        self.coin_name = coin_name
        self.lookback_window = lookback_window  # 과거 데이터 참조 윈도우 (과거 데이터만 참조하여 탐지하도록)
        self.prepare_data()
        
    def prepare_data(self):
        """데이터 전처리 및 특성 엔지니어링"""
        # timestamp를 datetime으로 변환 (UTC 기준)
        self.data['datetime'] = pd.to_datetime(self.data['timestamp'], unit='ms', utc=True)
        
        # 기본 가격 변동률 계산
        self.data['price_change'] = self.data['close'].pct_change()
        self.data['price_change_abs'] = abs(self.data['price_change'])
        
        # 가격 변동성 (high-low spread)
        self.data['hl_spread'] = (self.data['high'] - self.data['low']) / self.data['close']
        
        # 거래량 변화율
        self.data['volume_change'] = self.data['volume'].pct_change()
        self.data['volume_change_abs'] = abs(self.data['volume_change'])
        
        # 가격-거래량 곱 (거래대금)
        self.data['turnover'] = self.data['close'] * self.data['volume']
        self.data['turnover_change'] = self.data['turnover'].pct_change()
        
        # 롤링 통계량 계산 (5분, 15분, 30분 윈도우)
        for window in [5, 15, 30]:
            self.data[f'price_volatility_{window}m'] = self.data['price_change'].rolling(window).std()
            self.data[f'volume_ma_{window}m'] = self.data['volume'].rolling(window).mean()
            self.data[f'volume_std_{window}m'] = self.data['volume'].rolling(window).std()
        
        # 시간 관련 특성
        self.data['hour'] = self.data['datetime'].dt.hour
    
    def method1_statistical_outliers_rolling(self, columns=['price_change_abs', 'volume_change_abs'], 
                                           threshold_method='iqr', z_threshold=3, iqr_multiplier=1.5):
        """방법 1: 롤링 통계적 이상치 탐지 (Z-score, IQR) - 실시간 버전"""
        print("=" * 60)
        print(f"방법 1: 롤링 통계적 이상치 탐지 - {self.coin_name}")
        print("=" * 60)
        
        results = {}
        total_outliers = 0
        
        for col in columns:
            if col not in self.data.columns:
                continue
            
            outliers = pd.Series(False, index=self.data.index)
            outlier_scores = pd.Series(0.0, index=self.data.index)
            
            # 각 시점에서 과거 데이터만 사용하여 임계값 계산
            for i in range(self.lookback_window, len(self.data)):
                # 과거 lookback_window 기간의 데이터만 사용
                historical_data = self.data[col].iloc[i-self.lookback_window:i]
                current_value = self.data[col].iloc[i]
                
                if pd.isna(current_value) or len(historical_data.dropna()) < 10:
                    continue
                
                if threshold_method == 'zscore':
                    # Z-score 방법
                    hist_mean = historical_data.mean()
                    hist_std = historical_data.std()
                    if hist_std > 0:
                        z_score = abs(current_value - hist_mean) / hist_std
                        is_outlier = z_score > z_threshold
                        outlier_scores.iloc[i] = z_score
                    else:
                        is_outlier = False
                        
                elif threshold_method == 'iqr':
                    # IQR 방법
                    Q1 = historical_data.quantile(0.25)
                    Q3 = historical_data.quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower_bound = Q1 - iqr_multiplier * IQR
                        upper_bound = Q3 + iqr_multiplier * IQR
                        is_outlier = (current_value < lower_bound) or (current_value > upper_bound)
                        # IQR 기반 스코어 계산
                        if current_value > upper_bound:
                            outlier_scores.iloc[i] = (current_value - upper_bound) / IQR
                        elif current_value < lower_bound:
                            outlier_scores.iloc[i] = (lower_bound - current_value) / IQR
                    else:
                        is_outlier = False
                
                outliers.iloc[i] = is_outlier
            
            outlier_count = outliers.sum()
            outlier_percentage = (outlier_count / len(self.data)) * 100
            total_outliers += outlier_count
            
            # 결과 저장
            self.data[f'{col}_outlier'] = outliers
            self.data[f'{col}_outlier_score'] = outlier_scores
            
            results[col] = {
                'method': f"{threshold_method} (rolling)",
                'outliers': outliers,
                'count': outlier_count,
                'percentage': outlier_percentage,
                'scores': outlier_scores
            }
            
            print(f"\n📊 {col} 분석 결과 (Rolling {threshold_method}):")
            print(f"   - 이상치 개수: {outlier_count}개 ({outlier_percentage:.2f}%)")
            print(f"   - 분석 가능 구간: {len(self.data) - self.lookback_window}개 (전체 {len(self.data)}개)")
        
        results['total_outliers'] = total_outliers
        return results
    
    def method4_composite_anomaly_score_rolling(self, score_window=30):
        """방법 4: 롤링 복합 이상치 점수 계산"""
        print("\n" + "=" * 60)
        print(f"방법 4: 롤링 복합 이상치 점수 (실시간 모델용) - {self.coin_name}")
        print("=" * 60)
        
        features = ['price_change_abs', 'volume_change_abs', 'hl_spread']
        weights = {'price_change_abs': 0.4, 'volume_change_abs': 0.4, 'hl_spread': 0.2}
        
        composite_scores = pd.Series(0.0, index=self.data.index)
        
        # 각 시점에서 과거 데이터만 사용하여 정규화 및 점수 계산
        for i in range(self.lookback_window, len(self.data)):
            current_score = 0.0
            
            for feature in features:
                if feature not in self.data.columns:
                    continue
                
                # 과거 score_window 기간의 데이터로 정규화 기준 계산
                historical_data = self.data[feature].iloc[i-self.lookback_window:i]
                current_value = self.data[feature].iloc[i]
                
                if pd.isna(current_value) or len(historical_data.dropna()) < 10:
                    continue
                
                # 롤링 최소/최대값으로 0-1 정규화
                rolling_min = historical_data.min()
                rolling_max = historical_data.max()
                
                if rolling_max > rolling_min:
                    normalized = (current_value - rolling_min) / (rolling_max - rolling_min)
                else:
                    normalized = 0.0
                
                # 가중치 적용
                weighted_score = normalized * weights.get(feature, 0.0)
                current_score += weighted_score
            
            composite_scores.iloc[i] = current_score
        
        self.data['rolling_anomaly_score'] = composite_scores
        
        # 이상치 임계값 설정 (전체 계산된 점수의 상위 5%)
        valid_scores = composite_scores[composite_scores > 0]
        if len(valid_scores) > 0:
            anomaly_threshold = valid_scores.quantile(0.95)
            high_anomalies = composite_scores > anomaly_threshold
            high_anomaly_count = high_anomalies.sum()
        else:
            anomaly_threshold = 0
            high_anomaly_count = 0
        
        print(f"📊 롤링 복합 이상치 점수 분석:")
        print(f"   - 분석 가능 구간: {len(valid_scores)}개")
        print(f"   - 평균 점수: {valid_scores.mean():.4f}")
        print(f"   - 최대 점수: {valid_scores.max():.4f}")
        print(f"   - 이상치 임계값 (95th percentile): {anomaly_threshold:.4f}")
        print(f"   - 고위험 이상치: {high_anomaly_count}개 ({(high_anomaly_count/len(self.data)*100):.2f}%)")
        
        return composite_scores, anomaly_threshold, high_anomaly_count
    
    def real_time_anomaly_model(self, lookback_window=30):
        """개선된 실시간 이상 탐지 모델"""
        print("\n" + "=" * 60)
        print(f"개선된 실시간 이상 탐지 모델 - {self.coin_name}")
        print("=" * 60)
        
        anomaly_scores = pd.Series(0.0, index=self.data.index)
        anomaly_flags = pd.Series(False, index=self.data.index)
        
        for i in range(lookback_window, len(self.data)):
            # 현재 시점
            current = self.data.iloc[i]
            
            # 과거 lookback_window 기간의 기준값 계산
            historical_data = self.data.iloc[i-lookback_window:i]
            
            # 기준 통계량
            price_vol_mean = historical_data['price_change_abs'].mean()
            price_vol_std = historical_data['price_change_abs'].std()
            volume_mean = historical_data['volume'].mean()
            volume_std = historical_data['volume'].std()
            
            # 현재 값의 이상 정도 계산
            if price_vol_std > 0:
                price_anomaly_score = abs(current['price_change_abs'] - price_vol_mean) / price_vol_std
            else:
                price_anomaly_score = 0
                
            if volume_std > 0:
                volume_anomaly_score = abs(current['volume'] - volume_mean) / volume_std
            else:
                volume_anomaly_score = 0
            
            # 복합 점수 (0-1 스케일)
            composite_score = min(1.0, (price_anomaly_score * 0.6 + volume_anomaly_score * 0.4) / 5)
            
            # 이상치 플래그 (점수 > 0.7)
            is_anomaly = composite_score > 0.7
            
            anomaly_scores.iloc[i] = composite_score
            anomaly_flags.iloc[i] = is_anomaly
        
        # 결과 저장
        self.data['realtime_anomaly_score'] = anomaly_scores
        self.data['realtime_anomaly_flag'] = anomaly_flags
        
        # 통계
        valid_scores = anomaly_scores[anomaly_scores > 0]
        total_anomalies = anomaly_flags.sum()
        
        print(f"📊 실시간 모델 성능:")
        print(f"   - 분석 대상 기간: {len(valid_scores)}분")
        print(f"   - 탐지된 이상치: {total_anomalies}개 ({(total_anomalies/len(valid_scores)*100):.2f}%)")
        print(f"   - 평균 이상치 점수: {valid_scores.mean():.4f}")
        print(f"   - 최대 이상치 점수: {valid_scores.max():.4f}")
        
        return anomaly_scores, anomaly_flags, total_anomalies

    def get_latest_anomaly_score(self):
        """가장 최근의 이상치 점수 반환 (실시간 사용용)"""
        if 'realtime_anomaly_score' in self.data.columns:
            latest_score = self.data['realtime_anomaly_score'].iloc[-1]
            return latest_score
        return 0.0
    
    def get_anomaly_summary(self):
        """이상치 탐지 결과 요약 반환"""
        summary = {
            'coin_name': self.coin_name,
            'total_periods': len(self.data),
            'analysis_periods': len(self.data) - self.lookback_window,
        }
        
        if 'realtime_anomaly_flag' in self.data.columns:
            summary['total_anomalies'] = self.data['realtime_anomaly_flag'].sum()
            summary['anomaly_rate'] = summary['total_anomalies'] / summary['analysis_periods'] * 100
        
        if 'realtime_anomaly_score' in self.data.columns:
            valid_scores = self.data['realtime_anomaly_score'][self.data['realtime_anomaly_score'] > 0]
            summary['avg_score'] = valid_scores.mean()
            summary['max_score'] = valid_scores.max()
            summary['latest_score'] = self.data['realtime_anomaly_score'].iloc[-1]
        
        return summary


# 유틸리티 함수들
def load_crypto_data(file_path):
    """암호화폐 데이터 로드 함수"""
    try:
        df = pd.read_csv(file_path)
        # 필요한 컬럼만 선택
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_columns]
        return df
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        return None

def quick_anomaly_detection(file_path, coin_name, lookback_window=100):
    """빠른 이상치 탐지 함수"""
    df = load_crypto_data(file_path)
    if df is None:
        return None
    
    detector = CryptoAnomalyDetector(df, coin_name, lookback_window)
    
    # 핵심 분석만 실행
    result1 = detector.method1_statistical_outliers_rolling()
    composite_score, threshold, high_count = detector.method4_composite_anomaly_score_rolling()
    scores, flags, total = detector.real_time_anomaly_model()
    
    return detector, detector.get_anomaly_summary()

def print_ranking_summary(method_results):
    """각 방법별 이상치 탐지 순위 출력"""
    print("\n" + "=" * 80)
    print("🏆 실시간 이상치 탐지 방법별 코인 순위 (이상치 많이 발견된 순)")
    print("=" * 80)
    
    for method_name, results in method_results.items():
        print(f"\n📈 {method_name}")
        print("-" * 50)
        
        # 결과를 이상치 개수로 정렬
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (coin, count) in enumerate(sorted_results, 1):
            print(f"   {rank}위: {coin:8s} - {count:4d}개 이상치")