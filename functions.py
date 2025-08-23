import pandas as pd
import numpy as np
import mplfinance as mpf

# CSV 파일 로드 및 시간 필터링
def load_csv(path, start_date=None, end_date=None):
    """
    path: csv파일 경로
    start_date: "2025-08-01 15:30:00" or "2025-08-01"
    end_date: "2025-08-02 09:00:00" or "2025-08-02"
    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.set_index("timestamp", inplace=True)

    data_start = df.index.min()
    data_end = df.index.max()

    if start_date is not None:
        start_date = pd.to_datetime(start_date, utc=True)
        if start_date < data_start:
            raise ValueError(f"start_date({start_date}) < data_start({data_start})")
        df = df[df.index >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date, utc=True)
        if end_date > data_end:
            raise ValueError(f"end_date({end_date}) > data_end({data_end})")
        df = df[df.index <= end_date]

    return df


# rolling IQR 계산 (Q1, Q3, IQR)
def iqr_threshold(series, lb, shift):
    Q1 = series.rolling(lb).quantile(0.25).shift(shift)
    Q3 = series.rolling(lb).quantile(0.75).shift(shift)
    IQR = Q3 - Q1
    return Q1, Q3, IQR


# IQR 기반 이상탐지 (수익률, 거래량 기준)
def detect_anomalies_iqr(
    df: pd.DataFrame,
    lookback: int = 100,          # IQR 계산용 롤링 길이(분)
    k_ret: float = 1.5,          # 수익률 IQR 배수
    k_vol: float = 1.5,          # 거래량 IQR 배수
    use_log_return: bool = True, # 로그수익률 vs 수익률
    use_notional:bool = False,   # 거래량 vs 거래대금
    notional_min: float = 50_000,# 저유동성 필터(거래대금)
    price_col: str = "close",
    volume_col: str = "volume",
    shift: int = 10,
    score_scaling=True,
    return_volume_scoring_weight=2
):
    """
    df: time 정렬된 1분봉 DataFrame (columns에 close, volume 포함)
    반환: 원본 + IQR기준 is anomaly column + IQR 기준 이상 score column
    """
    out = df.copy()

    # 1) 수익률 계산
    if use_log_return:
        out["ret"] = np.log(out[price_col] / out[price_col].shift(1))
    else:
        out["ret"] = out[price_col].pct_change()

    if use_notional:
        out["qty"] = out[price_col] * out[volume_col]
        qty_name = "notional"
    else:
        out["qty"] = out[volume_col]
        qty_name = "volume"

    # 2) 거래대금(필터/참고용)
    out["notional"] = out[price_col] * out[volume_col]

    # 3) 이상상황 기준 설정
    # 이전 이상 상황 캔들 제외 위해 rough한 기준 계산
    f_r_Q1, f_r_Q3, f_r_IQR = iqr_threshold(out['ret'], lookback, shift=1)
    f_r_upper = f_r_Q3 + 1 * f_r_IQR
    f_r_lower = f_r_Q1 - 1 * f_r_IQR
    f_v_Q1, f_v_Q3, f_v_IQR = iqr_threshold(out['qty'], lookback, shift=1)
    f_v_upper = f_v_Q3 + 1 * f_v_IQR

    # 이전 이상 상황 캔들 제외
    # flag1_ret = (out['ret'] > f_r_upper) | (out['ret'] < f_r_lower)
    # flag1_vol = (out['qty'] > f_v_upper)
    # flag1 = (flag1_ret & flag1_vol).fillna(False)

    # ret_w = out['ret'].where(~flag1)

    # 위 코드로 하면 이상 상황으로 간주되는 캔들이 엄청 줄어서 (왜 그런지 디버깅 필요)
    # 일단 이상 상황 캔들 값들이 크게 영향 주지 못하도록 clip으로 대체
    ret_w = out['ret'].clip(lower=f_r_lower, upper=f_r_upper)

    # 수익률 IQR (가장 가까운 10개 캔들은 보지 않음)
    r_Q1, r_Q3, r_IQR = iqr_threshold(ret_w, lookback, shift=shift)

    r_lower = r_Q1 - k_ret * r_IQR
    r_upper = r_Q3 + k_ret * r_IQR

    # 거래량 IQR (가장 가까운 10개 캔들은 보지 않음)
    qty_w = out['qty'].clip(upper=f_v_upper)
    # qty_w = out['qty'].where(~flag1)
    v_Q1, v_Q3, v_IQR = iqr_threshold(qty_w, lookback, shift=shift)

    v_upper = v_Q3 + k_vol * v_IQR

    # 4) 이상 플래그
    # 수익률: 양/음 급변 모두 감지
    out["anomaly_ret"] = (out["ret"] < r_lower) | (out["ret"] > r_upper)

    # 거래량: 상방 급증 위주(원하면 | |로 하방도 포함 가능)
    out["anomaly_vol"] = (out["qty"] > v_upper)
    # out["anomaly_vol"] = (out[volume_col] < v_lower) | (out[volume_col] > v_upper)

    # 5) 정렬용 스코어 (임계 초과량을 IQR로 정규화)
    eps = 1e-12
    # 수익률 초과량 (양/음 모두)
    over_upper_r = np.maximum(out["ret"] - r_upper, 0.0)
    under_lower_r = np.maximum(r_lower - out["ret"], 0.0)
    if score_scaling:
        ret_excess = (over_upper_r + under_lower_r) / (r_IQR.abs() + eps)
    else:
        ret_excess = (over_upper_r + under_lower_r)

    # 거래량 초과량 (상방)
    if score_scaling:
        vol_excess = np.maximum(out["qty"] - v_upper, 0.0) / (v_IQR.abs() + eps)
    else:
        vol_excess = np.maximum(out["qty"] - v_upper, 0.0)

    # anomaly score - 2개 사이 weight 조절 필요 + high/low range같은 추가 변수 필요할지도
    out["score"] = ret_excess.fillna(0.0) * return_volume_scoring_weight + vol_excess.fillna(0.0)

    # 6) 유동성 필터 적용
    liquidity_ok = out["notional"] >= notional_min

    out["is_anomaly"] = (out["anomaly_ret"] & out["anomaly_vol"]) & liquidity_ok

    # 임계선이 NaN인 초기구간은 False/0 처리
    out["is_anomaly"] = out["is_anomaly"].fillna(False)
    out["score"] = out["score"].fillna(0.0)

    # 디버깅/모니터링에 유용한 컷 정보도 남김(선택)
    out["r_lower"], out["r_upper"] = r_lower, r_upper
    out["v_upper"] = v_upper  # 필요 시 v_lower도 추가

    return out


# 1차트 데이터(std 기반)에서 이상탐지 점수 계산
def onechart_score(min_df, onechart_df, std_count_scaling=False, shift=1, lookback=100, k=1.5):
    out = min_df.copy()

    # tick단위 1chart를 1분단위로 집계 [t, t+1)
    per_min_std = (onechart_df['fx_diff'].astype(float).resample("1min", label="left", closed="left").std()).reindex(out.index)
    # per_min_diff_std = (onechart_df['fx_diff'].astype(float).resample("1min", label="left", closed="left").diff().std()).reindex(out.index)
    per_min_tick_count = (onechart_df['fx_diff'].resample("1min", label="left", closed="left").count()).reindex(out.index)

    if std_count_scaling:
        per_min_std = per_min_std / np.sqrt(per_min_tick_count.astype(float))

    # IQR 정규화 (현재분 제외 위해 shift)
    base = per_min_std.shift(shift)
    Q1 = base.rolling(lookback, min_periods=lookback).quantile(0.25)
    Q3 = base.rolling(lookback, min_periods=lookback).quantile(0.75)
    IQR = Q3 - Q1

    upper = Q3 + k * IQR

    # 초과분 계산
    over = (per_min_std - upper).clip(lower=0)
    # IQR scaling (평소에 변동하던 변동폭 정도로 나누기)
    score = over / (IQR.abs() + 1e-12)

    out["1min_std"] = per_min_std
    out["1min_tick_count"] = per_min_tick_count
    out["onechart_score"] = score

    return out


# 이상탐지 결과 캔들차트 + scatter 표시
def plot_anomaly(df, idx, anomaly_key="is_anomaly"):
    """
    df: time 정렬된 1분봉 DataFrame (columns에 close, volume 포함)
    반환: 원본 + IQR기준 이상탐지 결과 컬럼들
    """
    score_points = np.where(df[anomaly_key] == 1, df['close'], np.nan)
    add_plots = [
        mpf.make_addplot(score_points[idx[0]:idx[1]], type='scatter', marker='o', markersize=10, color='blue'),
    ]
    mpf.plot(df.iloc[idx[0]:idx[1]], type='candle', style='yahoo', addplot=add_plots, volume=True)


# 이상 점수 기반 캔들차트 + scatter 표시
def plot_anomaly_score(df, idx, q, anomaly_key="score"):
    threshold = df[anomaly_key].quantile(q)
    score_points = np.where(df[anomaly_key] > threshold, df['close'], np.nan)
    add_plots = [
        mpf.make_addplot(score_points[idx[0]:idx[1]], type='scatter', marker='o', markersize=10, color='blue'),
    ]
    mpf.plot(df.iloc[idx[0]:idx[1]], type='candle', style='yahoo', addplot=add_plots, volume=True)
