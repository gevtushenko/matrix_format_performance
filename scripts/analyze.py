import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


path_to_results = '../results/mkl_ht_1_tb_1'


def setup_printer():
    sns.set()

    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.max_columns', 30)
    pd.set_option('expand_frame_repr', False)


def load_data(file):
    matrices_info = pd.read_json('{}/matrices_info.json'.format(path_to_results)).T
    source = pd.read_json(file).T
    merged = pd.merge(left=source, right=matrices_info, left_index=True, right_index=True)
    merged['nnzpr'] = merged['nnz'] / merged['rows']
    return source, merged


def calculate_speedup(merged, source):
    speedup = merged.copy()
    columns = list(source)

    for col in columns:
        speedup[col] = source['CPU CSR'] / source[col]

    return speedup


def print_stats(label, speedup, nlargest=3):
    print("Data for {} => ".format(label))
    print(speedup.nlargest(100, 'CPU CSR (mkl)'))
    #print(speedup.nlargest(nlargest, 'GPU COO')[['CPU CSR', 'GPU COO', 'nnz']])
    #print(speedup.nlargest(nlargest, 'GPU ELL')[['CPU CSR', 'GPU ELL', 'nnz']])
    #print(speedup.nlargest(nlargest, 'GPU HYBRID 0')[['CPU CSR', 'GPU HYBRID 0', 'nnz']])
    print(speedup.describe())


setup_printer()

source_df, merged_df = load_data('{}/float.json'.format(path_to_results))
float_speedup = calculate_speedup(merged_df, source_df)

source_df, merged_df = load_data('{}/double.json'.format(path_to_results))
double_speedup = calculate_speedup(merged_df, source_df)

min_nnz_to_compare = 50000
float_speedup = float_speedup[float_speedup['nnz'] > min_nnz_to_compare]
double_speedup = double_speedup[double_speedup['nnz'] > min_nnz_to_compare]

print_stats('float', float_speedup)
print_stats('double', double_speedup)

# sns.pairplot(speedup)

# for col in columns:
#     plt.hist(speedup[col], normed=False, alpha=0.5)

# sns.distplot(float_speedup['CPU CSR Parallel'])
# sns.distplot(float_speedup['CPU CSR (mkl)'])

sns.distplot(float_speedup['GPU CSR (vector)'])
sns.distplot(float_speedup['GPU CSR-Adaptive'])

sns.jointplot(data=float_speedup, x='nnzpr', y='GPU CSR', kind='reg')
sns.jointplot(data=float_speedup, x='nnzpr', y='GPU ELL', kind='reg')

plt.show()
