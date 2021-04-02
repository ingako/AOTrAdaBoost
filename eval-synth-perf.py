#!/usr/bin/env python3

import os
import sys
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pprint import PrettyPrinter

from scipy.stats import friedmanchisquare

benchmark_list = ["disable_transfer",
                  "no_boost",
                  "ozaboost",
                  "tradaboost",
                  "atradaboost"]

@dataclass
class Param:
    exp_code: str = ""
    kappa_window: int = 60
    transfer_match_lowerbound: float = 0.0
    least_transfer_warning_period_instances_length: int = 100
    num_diff_distr_instances: int = 300
    transfer_kappa_threshold: float = 0.4
    bbt_pool_size: int = 40
    gamma: float = 4


tree_case = \
    Param(
        exp_code= 'tree/case',
        least_transfer_warning_period_instances_length = 200,
        num_diff_distr_instances = 100,
        transfer_kappa_threshold = 0.3,
        bbt_pool_size = 40,
        gamma = 8.0
    )
params = [tree_case]

noise_agrawal_0_0 = \
    Param(
        exp_code= 'noise-0.0-0.0',
        least_transfer_warning_period_instances_length = 100,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.4,
        bbt_pool_size = 40,
        gamma = 4.0
    )

noise_agrawal_1_0 = \
    Param(
        exp_code= 'noise-0.1-0.0',
        least_transfer_warning_period_instances_length = 100,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.3,
        bbt_pool_size = 50,
        gamma = 1.0
    )

noise_agrawal_2_1 = \
    Param(
        exp_code= 'noise-0.2-0.1',
        least_transfer_warning_period_instances_length = 100,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.3,
        bbt_pool_size = 10,
        gamma = 4.0
    )


noise_agrawal_0_0_gradual = \
    Param(
        exp_code= 'noise-0.0-0.0-gradual',
        least_transfer_warning_period_instances_length = 150,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.2,
        bbt_pool_size = 30,
        gamma = 5.0
    )

noise_agrawal_1_0_gradual = \
    Param(
        exp_code= 'noise-0.1-0.0-gradual',
        least_transfer_warning_period_instances_length = 300,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.2,
        bbt_pool_size = 50,
        gamma = 1.0
    )

noise_agrawal_2_1_gradual = \
    Param(
        exp_code= 'noise-0.2-0.1-gradual',
        least_transfer_warning_period_instances_length = 400,
        num_diff_distr_instances = 200,
        transfer_kappa_threshold = 0.2,
        bbt_pool_size = 30,
        gamma = 6.0
    )

# noise_agrawal_2_1_case = \
#     Param(
#         exp_code= 'noise-0.2-0.1-case',
#         least_transfer_warning_period_instances_length = 100,
#         num_diff_distr_instances = 200,
#         transfer_kappa_threshold = 0.2,
#         bbt_pool_size = 50,
#         gamma = 8.0
#     )

noise_agrawal_0_1 = \
    Param(
        exp_code= 'noise-0.0-0.1',
        least_transfer_warning_period_instances_length = 100,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.3,
        bbt_pool_size = 50,
        gamma = 9.0
    )

noise_agrawal_1_2 = \
    Param(
        exp_code= 'noise-0.1-0.2',
        least_transfer_warning_period_instances_length = 100,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.4,
        bbt_pool_size = 20,
        gamma = 6.0
    )

noise_agrawal_0_1_gradual = \
    Param(
        exp_code= 'noise-0.0-0.1-gradual',
        least_transfer_warning_period_instances_length = 100,
        num_diff_distr_instances = 200,
        transfer_kappa_threshold = 0.4,
        bbt_pool_size = 50,
        gamma = 5.0
    )

noise_agrawal_1_2_gradual = \
    Param(
        exp_code= 'noise-0.1-0.2-gradual',
        least_transfer_warning_period_instances_length = 100,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.2,
        bbt_pool_size = 40,
        gamma = 7.0
    )


# params = [noise_agrawal_0_0, noise_agrawal_1_0, noise_agrawal_2_1,
#           noise_agrawal_0_0_gradual, noise_agrawal_1_0_gradual, noise_agrawal_2_1_gradual,
#           ]

# reverse noise transfer
# params = [noise_agrawal_0_1, noise_agrawal_1_2,
#           noise_agrawal_0_1_gradual, noise_agrawal_1_2_gradual]

####################################################################

# noise_tree_0_0 = \
#     Param(
#         exp_code= 'tree/noise-0.0-0.0',
#         least_transfer_warning_period_instances_length = 200,
#         num_diff_distr_instances = 100,
#         transfer_kappa_threshold = 0.3,
#         bbt_pool_size = 50,
#         gamma = 1.0
#     )
# 
# noise_tree_1_0 = \
#     Param(
#         exp_code= 'tree/noise-0.1-0.0',
#         least_transfer_warning_period_instances_length = 300,
#         num_diff_distr_instances = 300,
#         transfer_kappa_threshold = 0.2,
#         bbt_pool_size = 10,
#         gamma = 10.0
#     )
# 
# noise_tree_2_1 = \
#     Param(
#         exp_code= 'tree/noise-0.2-0.1',
#         least_transfer_warning_period_instances_length = 100,
#         num_diff_distr_instances = 300,
#         transfer_kappa_threshold = 0.2,
#         bbt_pool_size = 20,
#         gamma = 6.0
#     )
# 
# noise_tree_0_0_gradual = \
#     Param(
#         exp_code= 'tree/noise-0.0-0.0-gradual',
#         least_transfer_warning_period_instances_length = 300,
#         num_diff_distr_instances = 300,
#         transfer_kappa_threshold = 0.2,
#         bbt_pool_size = 20,
#         gamma = 2.0
#     )
# 
# 
# noise_tree_1_0_gradual = \
#     Param(
#         exp_code= 'tree/noise-0.1-0.0-gradual',
#         least_transfer_warning_period_instances_length = 300,
#         num_diff_distr_instances = 200,
#         transfer_kappa_threshold = 0.2,
#         bbt_pool_size = 20,
#         gamma = 2.0
#     )
# 
# noise_tree_2_1_gradual = \
#     Param(
#         exp_code= 'tree/noise-0.2-0.1-gradual',
#         least_transfer_warning_period_instances_length = 100,
#         num_diff_distr_instances = 300,
#         transfer_kappa_threshold = 0.2,
#         bbt_pool_size = 20,
#         gamma = 6.0
#     )
# 
# params = [noise_tree_0_0, noise_tree_1_0, noise_tree_2_1,
#           noise_tree_0_0_gradual, noise_tree_1_0_gradual, noise_tree_2_1_gradual]
# 
# ####################################################################
# tree reverse noise

noise_tree_0_1 = \
    Param(
        exp_code= 'tree/noise-0.0-0.1',
        least_transfer_warning_period_instances_length = 100,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.4,
        bbt_pool_size = 40,
        gamma = 10.0
    )

noise_tree_1_2 = \
    Param(
        exp_code= 'tree/noise-0.1-0.2',
        least_transfer_warning_period_instances_length = 300,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.2,
        bbt_pool_size = 30,
        gamma = 8.0
    )

noise_tree_0_1_gradual = \
    Param(
        exp_code= 'tree/noise-0.0-0.1-gradual',
        least_transfer_warning_period_instances_length = 100,
        num_diff_distr_instances = 200,
        transfer_kappa_threshold = 0.2,
        bbt_pool_size = 40,
        gamma = 1.0
    )

noise_tree_1_2_gradual = \
    Param(
        exp_code= 'tree/noise-0.1-0.2-gradual',
        least_transfer_warning_period_instances_length = 300,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.2,
        bbt_pool_size = 20,
        gamma = 7.0
    )

# params = [noise_tree_0_1, noise_tree_1_2,
#           noise_tree_0_1_gradual, noise_tree_1_2_gradual]



# ####################################################################
# partial
# 
# partial_agrawal_70_109 = \
#     Param(
#         exp_code= 'partial-70-109',
#         least_transfer_warning_period_instances_length = 100,
#         num_diff_distr_instances = 300,
#         transfer_kappa_threshold = 0.2,
#         bbt_pool_size = 50,
#         gamma = 2.0
#     )
# 
# partial_agrawal_109_70 = \
#     Param(
#         exp_code= 'partial-109-70',
#         least_transfer_warning_period_instances_length = 100,
#         num_diff_distr_instances = 200,
#         transfer_kappa_threshold = 0.1,
#         bbt_pool_size = 50,
#         gamma = 1.0
#     )
# 
# 
# partial_agrawal_70_109_gradual = \
#     Param(
#         exp_code= 'partial-70-109-gradual',
#         least_transfer_warning_period_instances_length = 100,
#         num_diff_distr_instances = 300,
#         transfer_kappa_threshold = 0.3,
#         bbt_pool_size = 50,
#         gamma = 3.0
#     )
# 
# 
# partial_agrawal_109_70_gradual = \
#     Param(
#         exp_code= 'partial-109-70-gradual',
#         least_transfer_warning_period_instances_length = 100,
#         num_diff_distr_instances = 200,
#         transfer_kappa_threshold = 0.1,
#         bbt_pool_size = 20,
#         gamma = 7.0
#     )
# 
# params = [partial_agrawal_70_109, partial_agrawal_109_70,
#           partial_agrawal_70_109_gradual, partial_agrawal_109_70_gradual]

####################################################################
# partial tree

# partial_tree_73_84 = \
#     Param(
#         exp_code= 'tree/partial-73-84',
#         least_transfer_warning_period_instances_length = 200,
#         num_diff_distr_instances = 100,
#         transfer_kappa_threshold = 0.3,
#         bbt_pool_size = 50,
#         gamma = 1.0
#     )
# 
# partial_tree_84_73 = \
#     Param(
#         exp_code= 'tree/partial-84-73',
#         least_transfer_warning_period_instances_length = 200,
#         num_diff_distr_instances = 100,
#         transfer_kappa_threshold = 0.3,
#         bbt_pool_size = 50,
#         gamma = 1.0
#     )
# 
# partial_tree_73_84_gradual = \
#     Param(
#         exp_code= 'tree/partial-73-84-gradual',
#         least_transfer_warning_period_instances_length = 200,
#         num_diff_distr_instances = 100,
#         transfer_kappa_threshold = 0.2,
#         bbt_pool_size = 50,
#         gamma = 3.0
#     )
# 
# # TODO
# partial_tree_84_73_gradual = \
#     Param(
#         exp_code= 'tree/partial-84-73-gradual',
#         least_transfer_warning_period_instances_length = 100,
#         num_diff_distr_instances = 200,
#         transfer_kappa_threshold = 0.1,
#         bbt_pool_size = 50,
#         gamma = 1.0
#     )
# 
# params = [partial_tree_73_84, partial_tree_73_84_gradual]

def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True


for p in params:
    print(f"{p.exp_code}")

    for i in range(len(benchmark_list)):
        boost_mode = benchmark_list[i]
        path = f"{p.exp_code}/{benchmark_list[i]}"

        if boost_mode == "no_boost":
            path = f"{path}/" \
                   f"{p.transfer_match_lowerbound}/" \
                   f"{p.kappa_window}/" \
                   f"{p.least_transfer_warning_period_instances_length}/" \
                   f"{p.num_diff_distr_instances}/" \
                   f"{p.transfer_kappa_threshold}/"
        elif boost_mode == "ozaboost" or boost_mode == "tradaboost":
            path = f"{path}/" \
                   f"{p.transfer_match_lowerbound}/" \
                   f"{p.kappa_window}/" \
                   f"{p.least_transfer_warning_period_instances_length}/" \
                   f"{p.num_diff_distr_instances}/" \
                   f"{p.transfer_kappa_threshold}/" \
                   f"{p.bbt_pool_size}/"
        elif boost_mode == "atradaboost":
            path = f"{path}/" \
                   f"{p.transfer_match_lowerbound}/" \
                   f"{p.kappa_window}/" \
                   f"{p.least_transfer_warning_period_instances_length}/" \
                   f"{p.num_diff_distr_instances}/" \
                   f"{p.transfer_kappa_threshold}/" \
                   f"{p.bbt_pool_size}/" \
                   f"{p.gamma}/"

        acc_list = []
        kappa_list = []
        acc_gain_list = []
        kappa_gain_list = []
        # acc_gain_per_drift_list = []
        time_list = []

        for seed in range(10):
            # if boost_mode != "disable_transfer":
            disable_output = f"{p.exp_code}/disable_transfer/{seed}/result-stream-1.csv"
            disable_df = pd.read_csv(disable_output)

            result_path = f"{path}/{seed}/result-stream-1.csv"

            benchmark_df = pd.read_csv(result_path)

            metrics = []

            acc_list.extend(benchmark_df["accuracy"].to_list())
            kappa_list.extend(benchmark_df["kappa"].to_list())
            time_list.append(benchmark_df["time"].iloc[-1])

            if boost_mode == "disable_transfer":
                acc_gain_list.append(0)
                kappa_gain_list.append(0)
                # acc_gain_per_drift_list.append(0)
            else:

                acc_gain = 0
                for i in range(0, len(benchmark_df)):
                    acc_gain += benchmark_df["accuracy"][i] - disable_df["accuracy"][i]
                # acc_gain_list.append(benchmark_df["accuracy"].sum() - disable_df["accuracy"].sum())
                acc_gain_list.append(acc_gain)

                kappa_gain_list.append(benchmark_df["kappa"].sum() - disable_df["kappa"].sum())
                # gain = 0
                # for row in range(81, 90):
                #     gain += benchmark_df["accuracy"].iloc[row] - disable_df["accuracy"].iloc[row]
                # acc_gain_per_drift_list.append(gain)

        acc = np.mean(acc_list) * 100
        acc_std = np.std(acc_list) * 100
        metrics.append(f"${acc:.2f}" + " \\pm " + f"{acc_std:.2f}$")

        kappa = np.mean(kappa_list) * 100
        kappa_std = np.std(kappa_list) * 100
        metrics.append(f"${kappa:.2f}" + " \\pm " + f"{kappa_std:.2f}$")

        if boost_mode == "disable_transfer":
            metrics.append('-')
            metrics.append('-')
        else:
            # acc_gain_per_drift = np.mean(acc_gain_per_drift_list)
            # acc_gain_per_drift_std = np.std(acc_gain_per_drift_list)
            # metrics.append(f"${acc_gain_per_drift:.2f}" + " \\pm " + f"{acc_gain_per_drift_std:.2f}$")

            acc_gain = np.mean(acc_gain_list) * 100
            acc_gain_std = np.std(acc_gain_list) * 100
            metrics.append(f"${round(acc_gain)}" + " \\pm " + f"{round(acc_gain_std)}$")

            kappa_gain = np.mean(kappa_gain_list) * 100
            kappa_gain_std = np.std(kappa_gain_list) * 100
            metrics.append(f"${round(kappa_gain)}" + " \\pm " + f"{round(kappa_gain_std)}$")


        time = np.mean(time_list)
        time_std = np.std(time_list)
        metrics.append(f"${time:.2f}" + " \\pm " + f"{time_std:.2f}$")

        print(" & ".join(metrics) + " \\\\")
