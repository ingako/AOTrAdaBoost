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
seed = sys.argv[2]
base_dir = f"{base_dir}/{exp_code}"

@dataclass
class Param:
    exp_code: str = "covtype"
    seed: int = 0
    kappa: float = 0
    instance_store_size: int = 5000
    eviction_interval: int = 1000000 # disabled
    least_transfer_warning_period_instances_length: int = 100
    num_diff_distr_instances: int = 300
    transfer_kappa_threshold: float = 0.0
    bbt_pool_size: int = 10
    transfer_gamma: float = 2.0


bike_params = Param(
        exp_code = exp_code,
        least_transfer_warning_period_instances_length = 100,
        num_diff_distr_instances = 350, # 300
        transfer_kappa_threshold = 0.1,
        bbt_pool_size = 10,
        transfer_gamma = 2.0
)

if exp_code == "covtype":
    p = covtype_params
elif exp_code == "sensor":
    p = sensor_params
elif exp_code[:6] == "insect":
    p = insect_params
elif exp_code[:4] == "tree":
    p = tree_params
elif exp_code[:7] == "agrawal":
    p = agrawal_params
elif exp_code[:4] == "bike":
    p = bike_params

gain_report_path = f"{base_dir}/gain-pro-report.txt"
disable_output = f"{base_dir}/disable_transfer/result-stream-1.csv"

disable_df = pd.read_csv(disable_output)

cur_data_dir = f"{base_dir}/atradaboost/"

print(f"evaluating {exp_code}...")
print("evaluating params...")

gain_report_out = open(gain_report_path, "w")
gain_report_out.write("param,reuse-param,lossy-win,pool_size,#instances,tradaboost-disable\n")
param_strs = ["least_transfer_warning_period_instances_length",
              "num_diff_distr_instances",
              "transfer_kappa_threshold",
              "bbt_pool_size",
              "gamma"]

def eval_tradaboost_output(cur_data_dir, param_values, gain_report_out):

    if len(param_values) != len(param_strs):
        # recurse
        params = [f for f in os.listdir(cur_data_dir) if os.path.isdir(os.path.join(cur_data_dir, f))]
        print(f"evaluating {params}...")
        for cur_param in params:
            param_values.append(cur_param)
            eval_tradaboost_output(f"{cur_data_dir}/{cur_param}", param_values, gain_report_out)
            param_values.pop()

    else:
        tradaboost_output = f"{cur_data_dir}/result-stream-1.csv"
        gain_output = f"{cur_data_dir}/gain.csv"
        with open(gain_output, "w") as out:

            disable_acc = disable_df["accuracy"]

            if is_empty_file(tradaboost_output):
                return

            tradaboost_df = pd.read_csv(tradaboost_output)
            tradaboost_acc = tradaboost_df["accuracy"]

            num_instances = tradaboost_df["count"]

            out.write("#count,seq,backtrack," \
                      "adapt_win,stability," \
                      "hybrid,noboost-disable-gain,tradaboost-disable-gain,tradaboost-noboost-gain\n")

            end = len(tradaboost_acc)

            tradaboost_disable_gain = 0

            for i in range(0, end):
                tradaboost_disable_gain += tradaboost_acc[i] - disable_acc[i]

                if i == (end - 1):
                    gain_report_out.write(f"{param_values[0]},{param_values[1]},"
                                          f"{param_values[2]},{param_values[3]},"
                                          f"{param_values[4]},"
                                          f"{num_instances[i]},"
                                          f"{tradaboost_disable_gain}\n")

eval_tradaboost_output(cur_data_dir, [], gain_report_out)

gain_report_out.close()
