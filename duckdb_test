import duckdb

df = duckdb.sql(f"""
    SELECT *
    FROM read_csv_auto('./candle_data/ETH_1chart.csv')
""").df()

print(df)