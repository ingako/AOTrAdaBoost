#!/usr/bin/env python3

import os
import sys
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass

def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True

base_dir = os.getcwd()
exp_code = sys.argv[1]
boost_mode = sys.argv[2]
base_dir = f"{base_dir}/{exp_code}"

gain_report_path = f"{base_dir}/gain-pro-report.txt"

disable_output = f"{base_dir}/disable_transfer/result-stream-1.csv"
disable_df = pd.read_csv(disable_output)
disable_acc = disable_df["accuracy"]

cur_data_dir = f"{base_dir}/{boost_mode}/"

if boost_mode == "no_boost":
    param_strs = ["kappa_window",
                  "least_transfer_warning_period_instances_length",
                  "num_diff_distr_instances",
                  "transfer_kappa_threshold"]
elif boost_mode == "ozaboost" or boost_mode == "tradaboost":
    param_strs = ["kappa_window",
                  "least_transfer_warning_period_instances_length",
                  "num_diff_distr_instances",
                  "transfer_kappa_threshold",
                  "bbt_pool_size"]
elif boost_mode == "atradaboost":
    param_strs = ["kappa_window",
                  "least_transfer_warning_period_instances_length",
                  "num_diff_distr_instances",
                  "transfer_kappa_threshold",
                  "bbt_pool_size",
                  "gamma"]
else:
    print("unsupported boost mode")
    exit(1)


gain_report_out = open(gain_report_path, "w")
result_line = param_strs.copy()
result_line.append("acc-gain\n")
gain_report_out.write(",".join(result_line))

print(f"evaluating {exp_code}...")
print("evaluating params...")


def eval_output(cur_data_dir, param_values, gain_report_out):

    if len(param_values) != len(param_strs):
        # recurse
        params = [f for f in os.listdir(cur_data_dir) if os.path.isdir(os.path.join(cur_data_dir, f))]
        for cur_param in params:
            param_values.append(cur_param)
            eval_output(f"{cur_data_dir}/{cur_param}", param_values, gain_report_out)
            param_values.pop()

    else:
        print(f"evaluating {param_values}...")

        tradaboost_output = f"{cur_data_dir}/result-stream-1.csv"
        gain_output = f"{cur_data_dir}/gain.csv"

        if is_empty_file(tradaboost_output):
            return

        tradaboost_df = pd.read_csv(tradaboost_output)
        tradaboost_acc = tradaboost_df["accuracy"]

        tradaboost_disable_gain = 0
        end = len(tradaboost_acc)
        # end = 150

        for i in range(0, end):
            tradaboost_disable_gain += tradaboost_acc[i] - disable_acc[i]

            if i == (end - 1):
                result_line = [f"{param_values[i]}" for i in range(len(param_strs))]
                result_line.append(f"{tradaboost_disable_gain}\n")
                gain_report_out.write(",".join(result_line))


eval_output(cur_data_dir, [], gain_report_out)
gain_report_out.close()
