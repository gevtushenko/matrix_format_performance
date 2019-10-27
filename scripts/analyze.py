import pandas as pd

source_df = pd.read_json('../results/float.json').T
speedup = source_df.copy()

columns = list(source_df)

for col in columns:
    speedup[col] = source_df['CPU CSR Parallel'] / source_df[col]

print(speedup)
