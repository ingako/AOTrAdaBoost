#!/usr/bin/env python3

import copy
import os
import sys
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pprint import PrettyPrinter

from scipy.stats import friedmanchisquare

benchmark_list = ["atradaboost"]

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

noise_agrawal_2_1 = \
    Param(
        exp_code= 'noise-0.2-0.1',
        least_transfer_warning_period_instances_length = 100,
        num_diff_distr_instances = 300,
        transfer_kappa_threshold = 0.3,
        bbt_pool_size = 10,
        gamma = 4.0
    )

def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True


def eval_sensitivity(p, v):
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
            metrics.append(f"${round(acc_gain)}$")

            kappa_gain = np.mean(kappa_gain_list) * 100
            kappa_gain_std = np.std(kappa_gain_list) * 100
            metrics.append(f"${round(kappa_gain)}$")


        time = np.mean(time_list)
        time_std = np.std(time_list)
        metrics.append(f"${time:.2f}" + " \\pm " + f"{time:.2f}$")

        print(f"{v} & "+ " & ".join(metrics) + "\\\\")

print("least")
for v in [100, 200, 300, 400]:
    params = copy.deepcopy(noise_agrawal_2_1)
    params.least_transfer_warning_period_instances_length = v
    eval_sensitivity(params, v)

print("num_instances")
for v in [100, 200, 300]:
    params = copy.deepcopy(noise_agrawal_2_1)
    params.num_diff_distr_instances = v
    eval_sensitivity(params, v)

print("kappa")
for v in [0.0, 0.1, 0.2, 0.3, 0.4]:
    params = copy.deepcopy(noise_agrawal_2_1)
    params.transfer_kappa_threshold = v
    eval_sensitivity(params, v)

print("pool size")
for v in [10, 20, 30, 40, 50]:
    params = copy.deepcopy(noise_agrawal_2_1)
    params.bbt_pool_size = v
    eval_sensitivity(params, v)

print("gamma")
for v in [2.0, 4.0, 6.0, 8.0]:
    params = copy.deepcopy(noise_agrawal_2_1)
    params.gamma = v
    eval_sensitivity(params, v)
