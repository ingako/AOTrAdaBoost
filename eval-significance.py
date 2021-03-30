#!/usr/bin/env python3

import os
import sys
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pprint import PrettyPrinter

from scipy.stats import friedmanchisquare

generators = [
    "noise-0.0-0.0",
    # "noise-0.1-0.0",
    "noise-0.2-0.1",
    "tree/noise-0.0-0.0",
    "tree/noise-0.1-0.0",
    # "tree/noise-0.2-0.1"
]

@dataclass
class Param:
    generator: str = "covtype"
    transfer_match_lowerbound: float = 0.0
    kappa_window: int = 60
    least_transfer_warning_period_instances_length: int = 100
    num_diff_distr_instances: int = 300
    transfer_kappa_threshold: float = 0.4
    bbt_pool_size: int = 40
    gamma: float = 4

benchmark_list = ["no_boost",
                  "ozaboost",
                  "tradaboost",
                  "atradaboost"]

benchmark_params_list = [
    Param(
        least_transfer_warning_period_instances_length = 100,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.4,
        bbt_pool_size = 40,
        gamma = 4.0
    ),

    # Param(
    #     least_transfer_warning_period_instances_length = 100,
    #     num_diff_distr_instances = 300,
    #     transfer_kappa_threshold = 0.3,
    #     bbt_pool_size = 50,
    #     gamma = 7.0
    # ),

    Param(
        least_transfer_warning_period_instances_length = 100,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.3,
        bbt_pool_size = 10,
        gamma = 4.0
    ),

    # tree
    Param(
        # exp_code= 'tree/noise-0.0-0.0',
        least_transfer_warning_period_instances_length = 200,
        num_diff_distr_instances = 100,
        transfer_kappa_threshold = 0.3,
        bbt_pool_size = 50,
        gamma = 1.0
    ),

    Param(
        # exp_code= 'tree/noise-0.1-0.0',
        least_transfer_warning_period_instances_length = 300,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.2,
        bbt_pool_size = 10,
        gamma = 10.0
    )
]

def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True

# def get_metrics(df, gain_per_drift, gain):
#     return [df["accuracy"].mean()*100, df["kappa"].mean()*100,
#             gain_per_drift * 100, gain*100,
#             df["time"].iloc[-1]/60]


metrics = []
for i in range(3): # 3 metrics: acc, acc-gain, runtime
    metrics.append([])
    for j in range(len(benchmark_list)):
        metrics[i].append([])

for i in range(len(generators)): # for each dataset
    generator = generators[i]

    for seed in range(10):
        disable_output = f"{generator}/disable_transfer/{seed}/result-stream-1.csv"
        disable_df = pd.read_csv(disable_output)
        p = benchmark_params_list[i]

        for j in range(len(benchmark_list)): # for each benchmark
            boost_mode = benchmark_list[j]
            path = f"{generator}/{benchmark_list[j]}"

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

            path = f"{path}/{seed}/result-stream-1.csv"

            benchmark_df = pd.read_csv(path)

            metrics[0][j].append(benchmark_df["accuracy"].mean())
            metrics[1][j].append(benchmark_df["accuracy"].sum()
                                     - disable_df["accuracy"].sum())
            metrics[2][j].append(benchmark_df["time"].iloc[-1]/60)

print("\nFriedman Test")
for i in range(3): # for each metric [acc, kappa, ...]
    stat, p = friedmanchisquare(metrics[i][0],
                                metrics[i][1],
                                metrics[i][2],
                                metrics[i][3])

    print('Statistics=%.3f, p=%.3f' % (stat, p))

    alpha = 0.05
    if p > alpha:
            print('Same distributions (fail to reject H0)')
    else:
            print('Different distributions (reject H0)')

