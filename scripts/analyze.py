import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

pd.set_option('display.max_colwidth', -1)
matrices_info = pd.read_json('../results/matrices_info.json').T
source_df = pd.read_json('../results/float.json').T
merged_df = pd.merge(left=source_df, right=matrices_info, left_index=True, right_index=True)
speedup = source_df.copy()

columns = list(source_df)

for col in columns:
    speedup[col] = source_df['CPU CSR'] / source_df[col]

print(speedup.nlargest(5, 'GPU CSR')[['CPU CSR', 'GPU CSR']])
print(speedup.nlargest(5, 'GPU COO')[['CPU CSR', 'GPU COO']])
print(speedup.nlargest(5, 'GPU ELL')[['CPU CSR', 'GPU ELL']])
print(speedup.nlargest(5, 'GPU HYBRID 0')[['CPU CSR', 'GPU HYBRID 0']])

speedup = speedup[speedup['CPU CSR Parallel'] > 1.0]

# sns.pairplot(speedup)

# for col in columns:
#     plt.hist(speedup[col], normed=False, alpha=0.5)

sns.distplot(speedup['CPU CSR Parallel'])
sns.distplot(speedup['GPU COO'])

plt.show()
