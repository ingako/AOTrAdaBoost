#!/usr/bin/env python3

import os
import sys
import math
import statistics as stats
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


noise_tree_0_0 = \
    Param(
        exp_code= 'tree/noise-0.0-0.0',
        least_transfer_warning_period_instances_length = 200,
        num_diff_distr_instances = 100,
        transfer_kappa_threshold = 0.3,
        bbt_pool_size = 50,
        gamma = 1.0
    )

noise_tree_1_0 = \
    Param(
        exp_code= 'tree/noise-0.1-0.0',
        least_transfer_warning_period_instances_length = 300,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.2,
        bbt_pool_size = 10,
        gamma = 10.0
    )


params = [noise_tree_0_0, noise_tree_1_0]

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
        gain_per_drift = []
        acc_gain_list = []
        time_list = []

        for seed in range(10):
            # if boost_mode != "disable_transfer":
            disable_output = f"{p.exp_code}/disable_transfer/{seed}/result-stream-1.csv"
            disable_df = pd.read_csv(disable_output)

            result_path = f"{path}/{seed}/result-stream-1.csv"

            benchmark_df = pd.read_csv(result_path)

            metrics = []
            acc_list.append(benchmark_df["accuracy"].mean())
            kappa_list.append(benchmark_df["kappa"].mean())
            time_list.append(benchmark_df["time"].iloc[-1]/60)

            if boost_mode == "disable_transfer":
                acc_gain_list.append(0)
            else:
                acc_gain_list.append(benchmark_df["accuracy"].sum() - disable_df["accuracy"].sum())

        acc = stats.mean(acc_list)
        acc_std = stats.stdev(acc_list)
        metrics.append(f"${acc:.2f}" + " \\pm " + f"{acc_std:.2f}$")

        kappa = stats.mean(kappa_list)
        kappa_std = stats.stdev(kappa_list)
        metrics.append(f"${kappa:.2f}" + " \\pm " + f"{kappa_std:.2f}$")

        if boost_mode == "disable_transfer":
            metrics.append('-')
        else:
            acc_gain = stats.mean(acc_gain_list)
            acc_gain_std = stats.stdev(acc_gain_list)
            metrics.append(f"${acc_gain:.2f}" + " \\pm " + f"{acc_gain_std:.2f}$")

        time = stats.mean(time_list)
        time_std = stats.stdev(time_list)
        metrics.append(f"${time:.2f}$")

        print(" & ".join(metrics))
