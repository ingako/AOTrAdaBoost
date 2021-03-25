#!/usr/bin/env python3

import sys
# path = r'../'
path = r'./cmake-build-debug/'

if path not in sys.path:
    sys.path.append(path)

from ctypes import *
# libc = cdll.LoadLibrary("cmake-build-debug/trans_pearl_wrapper.cpython-37m-darwin.so")
libc = cdll.LoadLibrary("cmake-build-debug/trans_pearl_wrapper.cpython-37m-x86_64-linux-gnu.so")

import argparse
import math
import random
import pathlib
import logging
import os.path
from collections import deque

import numpy as np

from evaluator import Evaluator
from trans_pearl_wrapper import adaptive_random_forest, pearl, trans_pearl_wrapper, trans_tree_wrapper

formatter = logging.Formatter('%(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # transfer learning params
    parser.add_argument("--transfer",
                        dest="transfer", action="store_true",
                        help="Enable transfer learning for PEARL")
    parser.set_defaults(transfer=False)
    parser.add_argument("--transfer_tree",
                        dest="transfer_tree", action="store_true",
                        help="Enable transfer learning for a single tree")
    parser.set_defaults(transfer=False)
    parser.add_argument("--transfer_streams_paths",
                        dest="transfer_streams_paths", default="", type=str,
                        help="Data stream paths for transfer learning")
    parser.add_argument("--exp_code",
                        dest="exp_code", default="", type=str,
                        help="Experiment code for result logging path")
    parser.add_argument("--least_transfer_warning_period_instances_length",
                        dest="least_transfer_warning_period_instances_length", default=50, type=int,
                        help="The least number of warning period instances needed to perform transfer learning")
    parser.add_argument("--instance_store_size",
                        dest="instance_store_size", default=8000, type=int,
                        help="Number of instances stored in each tree for generating pseudo data with KNN")
    parser.add_argument("--num_diff_distr_instances",
                        dest="num_diff_distr_instances", default=30, type=int,
                        help="number of instances from the matched tree to generate for transfer learning")
    parser.add_argument("--bbt_pool_size",
                        dest="bbt_pool_size", default=100, type=int,
                        help="The size of the background boosting tree pool")
    parser.add_argument("--eviction_interval",
                        dest="eviction_interval", default=1000000, type=int,
                        help="Eviction interval for boosting")
    parser.add_argument("--transfer_kappa_threshold",
                        dest="transfer_kappa_threshold", default=0.3, type=float,
                        help="Kappa threshold for swapping foregournd tree with transfer tree")
    parser.add_argument("--transfer_gamma",
                        dest="transfer_gamma", default=3, type=float,
                        help="Gamma for adaptive tradaboost")
    parser.add_argument("--boost_mode",
                        dest="boost_mode", default="otradaboost", type=str,
                        help="no_boost, ozaboost, tradaboost, otradaboost")

    # real world datasets
    parser.add_argument("--dataset_name",
                        dest="dataset_name", default="", type=str,
                        help="dataset name")
    parser.add_argument("--data_format",
                        dest="data_format", default="", type=str,
                        help="dataset format {csv|arff}")

    # pre-generated synthetic datasets
    parser.add_argument("-g", "--is_generated_data",
                        dest="is_generated_data", action="store_true",
                        help="Handle dataset as pre-generated synthetic dataset")
    parser.set_defaults(is_generator_data=False)
    parser.add_argument("--generator_name",
                        dest="generator_name", default="agrawal", type=str,
                        help="name of the synthetic data generator")
    parser.add_argument("--generator_traits",
                        dest="generator_traits", default="abrupt/poisson3", type=str,
                        help="Traits of the synthetic data")
    parser.add_argument("--generator_seed",
                        dest="generator_seed", default=0, type=int,
                        help="Seed used for generating synthetic data")

    # pearl params
    parser.add_argument("-t", "--tree",
                        dest="num_trees", default=60, type=int,
                        help="number of trees in the forest")
    parser.add_argument("-c", "--candidate_tree",
                        dest="max_num_candidate_trees", default=60, type=int,
                        help="max number of candidate trees in the forest")
    # parser.add_argument("--pool",
    #                     dest="tree_pool_size", default=180, type=int,
    #                     help="number of trees in the online tree repository")
    parser.add_argument("-w", "--warning",
                        dest="warning_delta", default=0.0001, type=float,
                        help="delta value for drift warning detector")
    parser.add_argument("-d", "--drift",
                        dest="drift_delta", default=0.00001, type=float,
                        help="delta value for drift detector")
    parser.add_argument("--drift_tension",
                        dest="drift_tension", default=-1.0, type=float,
                        help="delta value for drift tension")
    parser.add_argument("--max_samples",
                        dest="max_samples", default=200000, type=int,
                        help="total number of samples")
    parser.add_argument("--sample_freq",
                        dest="sample_freq", default=100, type=int,
                        help="log interval for performance")
    parser.add_argument("--kappa_window",
                        dest="kappa_window", default=50, type=int,
                        help="number of instances must be seen for calculating kappa")
    parser.add_argument("--poisson_lambda",
                        dest="poisson_lambda", default=6, type=int,
                        help="lambda for poisson distribution")
    parser.add_argument("--random_state",
                        dest="random_state", default=0, type=int,
                        help="Seed used for adaptive hoeffding tree")

    parser.add_argument("-s", "--enable_state_adaption",
                        dest="enable_state_adaption", action="store_true",
                        help="enable the state adaption algorithm")
    parser.set_defaults(enable_state_adaption=False)
    parser.add_argument("-p", "--enable_state_graph",
                        dest="enable_state_graph", action="store_true",
                        help="enable state transition graph")
    parser.set_defaults(enable_state_graph=False)

    parser.add_argument("--cd_kappa_threshold",
                        dest="cd_kappa_threshold", default=0.2, type=float,
                        help="Kappa value that the candidate tree needs to outperform both"
                             "background tree and foreground drifted tree")
    parser.add_argument("--bg_kappa_threshold",
                        dest="bg_kappa_threshold", default=0.00, type=float,
                        help="Kappa value that the background tree needs to outperform the "
                             "foreground drifted tree to prevent from false positive")
    parser.add_argument("--edit_distance_threshold",
                        dest="edit_distance_threshold", default=100, type=int,
                        help="The maximum edit distance threshold")
    parser.add_argument("--lossy_window_size",
                        dest="lossy_window_size", default=5, type=int,
                        help="Window size for lossy count")
    parser.add_argument("--reuse_window_size",
                        dest="reuse_window_size", default=0, type=int,
                        help="Window size for calculating reuse rate")
    parser.add_argument("--reuse_rate_upper_bound",
                        dest="reuse_rate_upper_bound", default=0.4, type=float,
                        help="The reuse rate threshold for switching from "
                             "pattern matching to graph transition")
    parser.add_argument("--reuse_rate_lower_bound",
                        dest="reuse_rate_lower_bound", default=0.1, type=float,
                        help="The reuse rate threshold for switching from "
                             "pattern matching to graph transition")

    args = parser.parse_args()

    # other params for arf/pearl
    arf_max_features = -1
    num_features = -1

    repo_size = args.num_trees * 1600
    np.random.seed(args.random_state)
    random.seed(0)

    if args.reuse_rate_upper_bound < args.reuse_rate_lower_bound:
        exit("reuse rate upper bound must be greater than or equal to the lower bound")

    if args.enable_state_graph:
        args.enable_state_adaption = True

    # prepare data
    data_file_path = args.transfer_streams_paths
    result_directory = f"{args.exp_code}/"

    # set result logging directory for all streams
    if args.transfer:
        result_directory = f"{result_directory}/transfer-pearl/" \
                           f"{args.least_transfer_warning_period_instances_length}/{args.instance_store_size}/" \
                           f"{args.num_diff_distr_instances}/" \
                           f"{args.transfer_kappa_threshold}/{args.bbt_pool_size}/" \
                           f"{args.boost_mode}/{args.generator_seed}/"

    elif args.transfer_tree:

        result_directory = f"{result_directory}/{args.boost_mode}/"
        if args.boost_mode == "disable_transfer":
            pass
        elif args.boost_mode == "no_boost":
            result_directory = f"{result_directory}/" \
                               f"{args.least_transfer_warning_period_instances_length}/" \
                               f"{args.num_diff_distr_instances}/" \
                               f"{args.transfer_kappa_threshold}/"
        elif args.boost_mode == "ozaboost" or args.boost_mode == "tradaboost":
            result_directory = f"{result_directory}/" \
                               f"{args.least_transfer_warning_period_instances_length}/" \
                               f"{args.num_diff_distr_instances}/" \
                               f"{args.transfer_kappa_threshold}/" \
                               f"{args.bbt_pool_size}/"
        elif args.boost_mode == "atradaboost":
            result_directory = f"{result_directory}/" \
                               f"{args.least_transfer_warning_period_instances_length}/" \
                               f"{args.num_diff_distr_instances}/" \
                               f"{args.transfer_kappa_threshold}/" \
                               f"{args.bbt_pool_size}/" \
                               f"{args.transfer_gamma}/"
        else:
            print("unsupported boost mode")
            exit(1)

        if args.is_generated_data:
            result_directory = f"{result_directory}/{args.generator_seed}/"

    else:
        result_directory = f"{result_directory}/pearl/{args.generator_seed}/"

    pathlib.Path(result_directory).mkdir(parents=True, exist_ok=True)

    # prepare metrics loggers for each stream
    metrics_loggers = []
    for idx in range(len(data_file_path.split(";"))):
        metric_output_file = f"{result_directory}/" \
                             f"result-stream-{idx}.csv"
        print(metric_output_file)
        metrics_logger = setup_logger(f'metrics-{idx}', metric_output_file)
        metrics_logger.info("count,accuracy,kappa,candidate_tree_size,transferred_tree_count,tree_pool_size,time")
        metrics_loggers.append(metrics_logger)

    # TODO
    acc_per_drift_logger = setup_logger('acc_per_drift', f'{result_directory}/acc-per-drift-{args.generator_seed}.log')

    expected_drift_locs_list = []
    if args.is_generated_data:
        for file_path in data_file_path.split(";"):
            if not args.is_generated_data:
                continue
            # for calculating acc per drift
            expected_drift_locs = deque()
            expected_drift_locs_log = f"{file_path}/drift-{args.generator_seed}.log"
            with open(f"{expected_drift_locs_log}", 'r') as f:
                for line in f:
                    expected_drift_locs.append(int(line))
            expected_drift_locs_list.append(expected_drift_locs)


    if args.transfer:
        classifier = trans_pearl_wrapper(len(data_file_path.split(";")),
                                         args.num_trees,
                                         args.max_num_candidate_trees,
                                         repo_size,
                                         args.edit_distance_threshold,
                                         args.kappa_window,
                                         args.lossy_window_size,
                                         args.reuse_window_size,
                                         arf_max_features,
                                         args.poisson_lambda,
                                         args.random_state,
                                         args.bg_kappa_threshold,
                                         args.cd_kappa_threshold,
                                         args.reuse_rate_upper_bound,
                                         args.warning_delta,
                                         args.drift_delta,
                                         args.least_transfer_warning_period_instances_length,
                                         args.instance_store_size,
                                         args.num_diff_distr_instances,
                                         args.bbt_pool_size,
                                         args.eviction_interval,
                                         args.transfer_kappa_threshold,
                                         args.boost_mode)

        # all_predicted_drift_locs, accepted_predicted_drift_locs = \

        # accepted_predicted_drifts_log_file = \
        #     f"{result_directory}/accepted-predicted-drifts-{args.generator_seed}.log"
        # all_predicted_drifts_log_file = \
        #     f"{result_directory}/all-predicted-drifts-{args.generator_seed}.log"

        # with open(accepted_predicted_drifts_log_file, "w") as accepted_f, \
        #         open(all_predicted_drifts_log_file, "w") as all_f:
        #     for i in range(args.num_trees):
        #         accepted_f.write(",".join([str(v) for v in accepted_predicted_drift_locs[i]]))
        #         accepted_f.write("\n")

        #         all_f.write(",".join([str(v) for v in all_predicted_drift_locs[i]]))
        #         all_f.write("\n")
    elif args.transfer_tree:
        classifier = trans_tree_wrapper(len(data_file_path.split(";")),
                                         args.random_state,
                                         args.kappa_window,
                                         args.warning_delta,
                                         args.drift_delta,
                                         args.least_transfer_warning_period_instances_length,
                                         args.instance_store_size,
                                         args.num_diff_distr_instances,
                                         args.bbt_pool_size,
                                         args.eviction_interval,
                                         args.transfer_kappa_threshold,
                                         args.transfer_gamma,
                                         args.boost_mode)


    else:
        # TODO
        print("main.py: init pearl")
        classifier = trans_pearl_wrapper(len(data_file_path.split(";")),
                      args.num_trees,
                      args.max_num_candidate_trees,
                      repo_size,
                      args.edit_distance_threshold,
                      args.kappa_window,
                      args.lossy_window_size,
                      args.reuse_window_size,
                      arf_max_features,
                      args.poisson_lambda,
                      args.random_state,
                      args.bg_kappa_threshold,
                      args.cd_kappa_threshold,
                      args.reuse_rate_upper_bound,
                      args.warning_delta,
                      args.drift_delta)

    # data_file_list = []
    # for file_path in data_file_path.split(";"):
    #     data_file_list.append(f'{file_path}/{args.generator_seed}.arff')
    data_file_list = data_file_path.split(";")

    if args.is_generated_data:
        for i in range(len(data_file_list)):
            data_file_list[i] = f"{data_file_list[i]}/{args.generator_seed}.arff"
            print(f"Preparing streams from files {data_file_list[i]}...")
            for file_path in data_file_path.split(";"):
                if not os.path.isfile(data_file_list[i]):
                    print(f"Cannot locate file at {data_file_list[i]}")
                    exit()


    evaluator = Evaluator()
    evaluator.prequential_evaluation_transfer(
        classifier=classifier,
        data_file_paths=data_file_list,
        max_samples=args.max_samples,
        sample_freq=args.sample_freq,
        metrics_loggers=metrics_loggers,
        expected_drift_locs_list=expected_drift_locs_list,
        acc_per_drift_logger=acc_per_drift_logger)

