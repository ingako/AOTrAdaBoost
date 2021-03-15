#!/usr/bin/env python3

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.rcParams["backend"] = "Qt4Agg"
# plt.rcParams["figure.figsize"] = (8, 5)
# plt.rcParams["figure.figsize"] = (6, 3)

# matplotlib.rcParams['figure.dpi'] = 280
# plt.rcParams["legend.loc"] = 'lower right'

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

kappa = 0.0
ed = 100

reuse_window = 0
reuse_rate = 0.9
lossy_window = 100000000

generator = "agrawal"
result_path = f"exp02-noise-0.2-0.1/"
# result_path = f"exp02-imbalance-19-91/"
# result_path = f"exp02/"

# fixed params
seed=0
eviction_interval = 1000000 # disabled
instance_store_size = 5000

least_transfer_warning_period_instances_length = 100 # [100 200 300]
num_diff_distr_instances = 10 # [10 20 30 40 50]
transfer_kappa_threshold = 0.3 # [0.1 0.2 0.3]
bbt_pool_size = 100 # [10]

boost_modes = ["disable_transfer", "no_boost", "ozaboost", "tradaboost", "otradaboost"]
avg_gain_results = [0, 0, 0, 0, 0]
avg_runtime_results = [0, 0, 0, 0, 0]

disable_transfer_acc = None

for idx, boost_mode in enumerate(boost_modes):

    exp_count = 0
    for least_transfer_warning_period_instances_length in [100, 200, 300]:
        for num_diff_distr_instances in [10, 20, 30, 40, 50]:
            for transfer_kappa_threshold in [0.1, 0.2, 0.3]:
                for bbt_pool_size in [10]:#, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    exp_count += 1

                    transfer_result_path = \
                        f"{result_path}/transfer-tree/" \
                        f"{least_transfer_warning_period_instances_length}/{instance_store_size}/" \
                        f"{num_diff_distr_instances}/{eviction_interval}/{transfer_kappa_threshold}/{bbt_pool_size}/"

                    stream_2_path = f"{transfer_result_path}/{boost_mode}/{seed}/result-stream-1.csv"
                    stream_2 = pd.read_csv(stream_2_path, index_col=0)
                    stream_2_acc = stream_2["accuracy"]

                    if boost_mode == "disable_transfer":
                        disable_transfer_acc = stream_2["accuracy"]

                    gain = 0
                    for row in range(int(20000/500)):
                        gain += stream_2_acc.iloc[row] - disable_transfer_acc.iloc[row]
                    avg_gain_results[idx] += gain
                    avg_runtime_results[idx] += stream_2["time"].iloc[int(20000/500)]

    avg_gain_results[idx] /= exp_count
    avg_runtime_results[idx] /= exp_count

print(avg_gain_results)
print(avg_runtime_results)
