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

result_directory_prefix = f"bike-bk/bike-weekday-weekend-500"

seed=0
least_transfer_warning_period_instances_length = 300
num_diff_distr_instances = 200
transfer_kappa_threshold = 0.1
bbt_pool_size = 40
transfer_gamma = 8.0

transfer_match_lowerbound = 0.0
kappa_window = 60

# stream_1_path = f"{transfer_result_path}/no_boost/{seed}/result-stream-0.csv"
# stream_1 = pd.read_csv(stream_1_path, index_col=0)
# plt.plot(stream_1["accuracy"], label="stream_1", linestyle="--")

for boost_mode in ["disable_transfer", "no_boost", "ozaboost", "tradaboost", "atradaboost"]:
    result_directory = f"{result_directory_prefix}/{boost_mode}/"
    if boost_mode == "disable_transfer":
        boost_mode_label = "HT"
        linestyle = '-'
    elif boost_mode == "no_boost":
        result_directory = f"{result_directory}/" \
                           f"{transfer_match_lowerbound}/" \
                           f"{kappa_window}/" \
                           f"{least_transfer_warning_period_instances_length}/{num_diff_distr_instances}/" \
                           f"{transfer_kappa_threshold}/"
        boost_mode_label = "w/o Boost"
        linestyle = '-'
    elif boost_mode == "ozaboost" or boost_mode == "tradaboost":
        result_directory = f"{result_directory}/" \
                           f"{transfer_match_lowerbound}/" \
                           f"{kappa_window}/" \
                           f"{least_transfer_warning_period_instances_length}/{num_diff_distr_instances}/" \
                           f"{transfer_kappa_threshold}/" \
                           f"{bbt_pool_size}/"
        if boost_mode == "ozaboost":
            boost_mode_label = "OzaBoost"
            linestyle = '--'
        if boost_mode == "tradaboost":
            boost_mode_label = "TrAdaBoost"
            linestyle = '-.'
    elif boost_mode == "atradaboost":
        result_directory = f"{result_directory}/" \
                           f"{transfer_match_lowerbound}/" \
                           f"{kappa_window}/" \
                           f"{least_transfer_warning_period_instances_length}/{num_diff_distr_instances}/" \
                           f"{transfer_kappa_threshold}/" \
                           f"{bbt_pool_size}/" \
                           f"{transfer_gamma}/"
        boost_mode_label = "AOTrAdaBoost"
        linestyle = ':'
    else:
        print("unsupported boost mode")
        exit(1)
    stream_2_path = f"{result_directory}/result-stream-1.csv"

    stream_2 = pd.read_csv(stream_2_path, index_col=0)
    plt.plot(stream_2["accuracy"], label=f"{boost_mode_label}", linestyle='-.')
    # print(stream_2["time"].iloc[-1]/60)


plt.legend()
plt.xlabel("No. of Instances")
plt.ylabel("Accuracy")

# plt.show()
plt.savefig('bike-result.pdf', bbox_inches='tight', dpi=300)
