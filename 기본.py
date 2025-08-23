import duckdb

df = duckdb.sql(f"""
                select *
                from read_csv_auto('./Anomaly_Detection/candle_data/YFI_1chart.csv')
                """).df()

print(df)