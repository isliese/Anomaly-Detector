from functions import *
import matplotlib

matplotlib.use('TkAgg',force=True)
print("Switched to:",matplotlib.get_backend())

path_1min = './Anomaly_Detection/candle_data/YFI.csv'
path_1chart = './Anomaly_Detection/candle_data/YFI_1chart.csv'

df = load_csv(path_1min)
df_1chart = load_csv(path_1chart)

df_iqr = detect_anomalies_iqr(df.copy(), lookback=100)
threshold = df_iqr["score"].quantile(0.95)
mask = df_iqr["score"] > threshold
plot_anomaly_score(df_iqr, idx= [100, 300], q=0.95)

df_score = onechart_score(df_iqr, df_1chart)

print(df_score["onechart_score"].mean())
print(df_score[mask]["onechart_score"].mean())

