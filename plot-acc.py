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
result_directory_prefix = f"bike-weekday-weekend"
# result_directory_prefix = f"exp02-noise-0-0/"
# result_directory_prefix = f"exp02-noise-0.1-0.1/"
# result_directory_prefix = f"exp02-noise-0.2-0.1/"
# result_directory_prefix = f"exp02-imbalance-19-91/"

seed=0
least_transfer_warning_period_instances_length = 100 # [100 200 300]
num_diff_distr_instances = 300 # [10 20 30 40 50]
transfer_kappa_threshold = 0.1 # [0.1 0.2 0.3]
bbt_pool_size = 10 # [10]
transfer_gamma = 2.0

# defaults
eviction_interval = 1000000 # disabled
instance_store_size = 5000 # default to 5000


# stream_1_path = f"{transfer_result_path}/no_boost/{seed}/result-stream-0.csv"
# stream_1 = pd.read_csv(stream_1_path, index_col=0)
# plt.plot(stream_1["accuracy"], label="stream_1", linestyle="--")

# pearl benchmark
# pearl_stream_2_path = f"{result_directory_prefix}/pearl/{seed}/result-stream-1.csv"
# pearl_stream_2 = pd.read_csv(pearl_stream_2_path, index_col=0)
# plt.plot(pearl_stream_2["accuracy"], label="pearl", linestyle="--")

for boost_mode in ["disable_transfer", "no_boost", "ozaboost", "tradaboost", "atradaboost"]:
    result_directory = f"{result_directory_prefix}/{boost_mode}/"
    if boost_mode == "disable_transfer":
        pass
    elif boost_mode == "no_boost":
        result_directory = f"{result_directory}/" \
                           f"{least_transfer_warning_period_instances_length}/{instance_store_size}/" \
                           f"{transfer_kappa_threshold}/"
    elif boost_mode == "ozaboost" or boost_mode == "tradaboost":
        result_directory = f"{result_directory}/" \
                           f"{least_transfer_warning_period_instances_length}/{instance_store_size}/" \
                           f"{transfer_kappa_threshold}/" \
                           f"{eviction_interval}/{num_diff_distr_instances}/{bbt_pool_size}/"
    elif boost_mode == "atradaboost":
        result_directory = f"{result_directory}/" \
                           f"{least_transfer_warning_period_instances_length}/{instance_store_size}/" \
                           f"{transfer_kappa_threshold}/" \
                           f"{eviction_interval}/{num_diff_distr_instances}/{bbt_pool_size}/" \
                           f"{transfer_gamma}/"
    else:
        print("unsupported boost mode")
        exit(1)
    stream_2_path = f"{result_directory}/result-stream-1.csv"

    stream_2 = pd.read_csv(stream_2_path, index_col=0)
    plt.plot(stream_2["accuracy"], label=f"stream_2 {boost_mode}", linestyle='-.')
    # print(stream_2["time"].iloc[-1]/60)


plt.legend()
plt.xlabel("No. of Instances")
plt.ylabel("Accuracy")

plt.show()
# plt.savefig('bike-result.png', bbox_inches='tight', dpi=300)
