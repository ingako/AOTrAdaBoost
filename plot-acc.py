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
generator_traits = "gradual"

arf_result_path = f"{generator}/{generator_traits}"
result_path = f"{generator}/{generator_traits}/k{kappa}-e{ed}/"

least_transfer_warning_period_instances_length = 50
instance_store_size = 500
num_pseudo_instances = 300
bbt_pool_size = 100
mini_batch_size = 100

result_path = \
    f"{result_path}/transfer/" \
    f"{least_transfer_warning_period_instances_length}/{instance_store_size}/" \
    f"{num_pseudo_instances}/{bbt_pool_size}/{mini_batch_size}/"

stream_1_path = f"{result_path}/result-0-stream-0.csv"
stream_2_path = f"{result_path}/result-0-stream-1.csv"

stream_1 = pd.read_csv(stream_1_path, index_col=0)
stream_2 = pd.read_csv(stream_2_path, index_col=0)

plt.plot(stream_1["accuracy"], label="stream_1", linestyle="--")
plt.plot(stream_2["accuracy"], label="stream_2", linestyle='-.')

plt.legend()
plt.xlabel("No. of Instances")
plt.ylabel("Accuracy")

plt.show()
