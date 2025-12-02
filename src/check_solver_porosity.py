import pandas as pd

df = pd.read_csv("results/flow_metrics/flow_metrics.csv")
print("Columns:", list(df.columns))
print(df.head())

if "Type" in df.columns and "Porosity" in df.columns:
    print("\nPorosity by Type:")
    print(df.groupby("Type")["Porosity"].describe())
else:
    print("\nNo 'Type' or 'Porosity' column found; got:", df.columns.tolist())
