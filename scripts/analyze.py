import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_printer():
    sns.set()

    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.max_columns', 30)
    pd.set_option('expand_frame_repr', False)


def load_data():
    matrices_info = pd.read_json('../results/matrices_info.json').T
    source = pd.read_json('../results/float.json').T
    merged = pd.merge(left=source, right=matrices_info, left_index=True, right_index=True)
    merged['nnzpr'] = merged['nnz'] / merged['rows']
    return source, merged


def calculate_speedup(merged, source):
    speedup = merged.copy()
    columns = list(source)

    for col in columns:
        speedup[col] = source['CPU CSR'] / source[col]

    return speedup


setup_printer()
source_df, merged_df = load_data()
speedup = calculate_speedup(merged_df, source_df)


print(speedup.nlargest(5, 'GPU CSR')[['CPU CSR', 'GPU CSR', 'nnz']])
print(speedup.nlargest(5, 'GPU COO')[['CPU CSR', 'GPU COO', 'nnz']])
print(speedup.nlargest(5, 'GPU ELL')[['CPU CSR', 'GPU ELL', 'nnz']])
print(speedup.nlargest(5, 'GPU HYBRID 0')[['CPU CSR', 'GPU HYBRID 0', 'nnz']])

# speedup = speedup[speedup['CPU CSR Parallel'] > 1.0]

# sns.pairplot(speedup)

# for col in columns:
#     plt.hist(speedup[col], normed=False, alpha=0.5)

# sns.distplot(speedup['CPU CSR Parallel'])
# sns.distplot(speedup['GPU COO'])


sns.jointplot(data=speedup, x='nnz', y='GPU CSR', kind='reg')
sns.jointplot(data=speedup, x='nnz', y='GPU ELL', kind='reg')

plt.show()
