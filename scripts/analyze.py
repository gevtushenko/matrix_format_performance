import pandas as pd

pd.set_option('display.max_colwidth', -1)
source_df = pd.read_json('../results/float.json').T
speedup = source_df.copy()

columns = list(source_df)

for col in columns:
    speedup[col] = source_df['CPU CSR'] / source_df[col]

print(speedup.nlargest(5, 'GPU CSR')[['CPU CSR', 'GPU CSR']])
print(speedup.nlargest(5, 'GPU COO')[['CPU CSR', 'GPU COO']])
print(speedup.nlargest(5, 'GPU ELL')[['CPU CSR', 'GPU ELL']])
