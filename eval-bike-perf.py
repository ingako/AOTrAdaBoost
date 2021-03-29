#!/usr/bin/env python3

import os
import sys
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pprint import PrettyPrinter

from scipy.stats import friedmanchisquare

generator = "bike-bk/bike-weekday-weekend"
benchmark_list = ["disable_transfer",
                  "no_boost",
                  "ozaboost",
                  "tradaboost",
                  "atradaboost"]


@dataclass
class Param:
    kappa_window: int = 60
    transfer_match_lowerbound: float = 0.0
    least_transfer_warning_period_instances_length: int = 100
    num_diff_distr_instances: int = 300
    transfer_kappa_threshold: float = 0.4
    bbt_pool_size: int = 40
    gamma: float = 4

p = \
    Param(
        least_transfer_warning_period_instances_length = 300,
        num_diff_distr_instances = 200,
        transfer_kappa_threshold = 0.1,
        bbt_pool_size = 40,
        gamma = 8.0
    )

def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True


disable_output = f"{generator}/disable_transfer/result-stream-1.csv"
disable_df = pd.read_csv(disable_output)

for i in range(len(benchmark_list)):
    boost_mode = benchmark_list[i]
    path = f"{generator}/{benchmark_list[i]}"

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

    path = f"{path}/result-stream-1.csv"

    benchmark_df = pd.read_csv(path)

    metrics = []
    acc = benchmark_df["accuracy"].mean()
    acc_std = benchmark_df["accuracy"].std()
    metrics.append(f"${acc:.2f}" + " \\pm " + f"{acc_std:.2f}$")

    kappa = benchmark_df["kappa"].mean()
    kappa_std = benchmark_df["kappa"].std()
    metrics.append(f"${kappa:.2f}" + " \\pm " + f"{kappa_std:.2f}$")

    if boost_mode == "disable_transfer":
        metrics.append('-')
        metrics.append('-')
    else:
        acc_gain = benchmark_df["accuracy"].sum() - disable_df["accuracy"].sum()
        metrics.append(f"${acc_gain:.2f}$")

        kappa_gain = benchmark_df["kappa"].sum() - disable_df["kappa"].sum()
        metrics.append(f"${kappa_gain:.2f}$")

    time = benchmark_df["time"].iloc[-1]/60
    metrics.append(f"${time:.2f}$")

    print(" & ".join(metrics))
