#!/usr/bin/env python3

import argparse
import math
import random
import pathlib
import logging
import os.path
from collections import deque

import numpy as np

from evaluator import Evaluator

import sys
path = r'../'

if path not in sys.path:
    sys.path.append(path)

# from build.trans_pearl import adaptive_random_forest, pearl, trans_pearl
# from trans_pearl import adaptive_random_forest, pearl, trans_pearl
from trans_pearl_wrapper import adaptive_random_forest, pearl, trans_pearl_wrapper
# from trans_pearl_wrapper import trans_pearl_wrapper

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

    # propearl specific params
    parser.add_argument("--transfer",
                        dest="transfer", action="store_true",
                        help="Enable ProPearl")
    parser.set_defaults(transfer=False)
    parser.add_argument("--pro_drift_window",
                        dest="pro_drift_window", default=100, type=int,
                        help="number of instances must be seen for proactive drift \
                        adaption")
    parser.add_argument("--hybrid",
                        dest="hybrid_delta", default=0.001, type=float,
                        help="delta value for proactive hybrid hoeffding bound")
    parser.add_argument("--backtrack_window",
                        dest="backtrack_window", default=25, type=int,
                        help="number of instances per eval when backtracking")
    parser.add_argument("--stability",
                        dest="stability_delta", default=0.001, type=float,
                        help="delta value for detecting stability")
    parser.add_argument("--sequence_len",
                        dest="sequence_len", default=8, type=int,
                        help="sequence length for sequence predictor")

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

    # transfer learning params
    parser.add_argument("--transfer_streams",
                        dest="transfer_streams", default="", type=str,
                        help="stream prefix for transfer learning")

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
                        dest="sample_freq", default=1000, type=int,
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

    if args.reuse_rate_upper_bound < args.reuse_rate_lower_bound:
        exit("reuse rate upper bound must be greater than or equal to the lower bound")

    if args.enable_state_graph:
        args.enable_state_adaption = True

    # prepare data
    if args.is_generated_data:
        data_file_dir = f"data/{args.generator_name}/" \
                        f"{args.generator_traits}/"

        data_file_path = ""
        if args.transfer:
            for seed in args.transfer_streams.split(";"):
                print(f"seed: {seed}")
                data_file_path += f"{data_file_dir}/{seed}.{args.data_format};"
            data_file_path = data_file_path[:-1]
            print(f"data_file_path: {data_file_path}")
            # e.g /root/transfer/data/agrawal/abrupt/0.arff;/root/transfer/data/agrawal/abrupt/0.arff"

        else:
            data_file_path = f"{data_file_dir}/{args.generator_seed}.{args.data_format}"
        result_directory = f"{args.generator_name}/{args.generator_traits}/"

    else:
        data_file_dir = f"../data/" \
                         f"{args.dataset_name}/"
        data_file_path = f"{data_file_dir}/{args.dataset_name}.{args.data_format}"
        result_directory = args.dataset_name

    for file_path in data_file_path.split(";"):
        if not os.path.isfile(file_path):
            print(f"Cannot locate file at {file_path}")
            exit()

    print(f"Preparing stream from file {data_file_path}...")


    if args.enable_state_graph:
        result_directory = f"{result_directory}/" \
                           f"k{args.cd_kappa_threshold}-e{args.edit_distance_threshold}/" \
                           f"r{args.reuse_rate_upper_bound}-r{args.reuse_rate_lower_bound}-" \
                           f"w{args.reuse_window_size}/" \
                           f"lossy-{args.lossy_window_size}"

    elif args.enable_state_adaption:
        result_directory = f"{result_directory}/" \
                           f"k{args.cd_kappa_threshold}-e{args.edit_distance_threshold}/"

    if args.transfer:
        result_directory = f"{result_directory}/transfer/" \
                           f"{args.sequence_len}/{args.backtrack_window}/" \
                           f"{args.pro_drift_window}/{args.stability_delta}/{args.hybrid_delta}"

    pathlib.Path(result_directory).mkdir(parents=True, exist_ok=True)

    metric_output_file = "result"
    if args.transfer:
        metric_output_file = f"{result_directory}/" \
                             f"{metric_output_file}-pro-{args.generator_seed}-{args.poisson_lambda}.csv"
    else:
        metric_output_file = f"{result_directory}/" \
                             f"{metric_output_file}-{args.generator_seed}-{args.poisson_lambda}.csv"


    configs = (
        f"metric_output_file: {metric_output_file}\n"
        f"warning_delta: {args.warning_delta}\n"
        f"drift_delta: {args.drift_delta}\n"
        f"max_samples: {args.max_samples}\n"
        f"sample_freq: {args.sample_freq}\n"
        f"kappa_window: {args.kappa_window}\n"
        f"random_state: {args.random_state}\n"
        f"enable_state_adaption: {args.enable_state_adaption}\n"
        f"enable_state_graph: {args.enable_state_graph}\n")

    print(configs)
    with open(f"{result_directory}/config", 'w') as out:
        out.write(configs)
        out.flush()

    # other params for pearl/propearl
    arf_max_features = -1
    num_features = -1

    # repo_size = args.num_trees * 160
    repo_size = args.num_trees * 1600
    np.random.seed(args.random_state)
    random.seed(0)

    if args.enable_state_adaption:
        with open(f"{result_directory}/reuse-rate-{args.generator_seed}.log", 'w') as out:
            out.write("background_window_count,candidate_window_count,reuse_rate\n")

    metrics_logger = setup_logger('metrics', metric_output_file)
    metrics_logger.info("count,accuracy,kappa,candidate_tree_size,tree_pool_size,time")

    process_logger = setup_logger('process', f'{result_directory}/processes-{args.generator_seed}.info')
    seq_logger = setup_logger('seq', f'{result_directory}/seq-pro-{args.generator_seed}.log')

    acc_per_drift_logger = setup_logger('acc_per_drift', f'{result_directory}/acc-per-drift-{args.generator_seed}.log')


    expected_drift_locs = None
    if args.is_generated_data:
        # for calculating acc per drift
        expected_drift_locs = deque()
        expected_drift_locs_log = f"{data_file_dir}/drift-{args.generator_seed}.log"
        with open(f"{expected_drift_locs_log}", 'r') as f:
            for line in f:
                expected_drift_locs.append(int(line))

    if args.transfer:
        stream_sequences_file_path = f"{data_file_dir}/sequence.txt"


    if not args.enable_state_adaption and not args.enable_state_graph:
        print("init adaptive_random_forest")
        pearl = adaptive_random_forest(args.num_trees,
                                       arf_max_features,
                                       args.poisson_lambda,
                                       args.random_state,
                                       args.warning_delta,
                                       args.drift_delta)
        eval_func = Evaluator.prequential_evaluation

        Evaluator.prequential_evaluation(
                classifier=pearl,
                stream=data_file_path,
                max_samples=args.max_samples,
                sample_freq=args.sample_freq,
                metrics_logger=metrics_logger,
                expected_drift_locs=expected_drift_locs,
                acc_per_drift_logger=acc_per_drift_logger)
    else:
        if args.transfer:
            stream_sequences = deque()

            with open(f"{stream_sequences_file_path}", 'r') as f:
                for line in f:
                    stream_sequences.append([int(v) for v in line.split()])

            metrics_loggers = []
            for idx in range(len(data_file_path.split(";"))):
                metric_output_file = f"{result_directory}/" \
                                     f"result-{args.generator_seed}-stream-{idx}.csv"
                print(metric_output_file)
                metrics_logger = setup_logger(f'metrics-{idx}', metric_output_file)
                metrics_logger.info("count,accuracy,kappa,candidate_tree_size,tree_pool_size,time")
                metrics_loggers.append(metrics_logger)

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
                                             args.pro_drift_window,
                                             args.hybrid_delta,
                                             args.backtrack_window,
                                             args.stability_delta)

            # all_predicted_drift_locs, accepted_predicted_drift_locs = \
            evaluator = Evaluator()
            evaluator.prequential_evaluation_transfer(
                classifier=classifier,
                data_file_paths=data_file_path.split(";"),
                max_samples=args.max_samples,
                sample_freq=args.sample_freq,
                metrics_loggers=metrics_loggers,
                seq_logger=seq_logger,
                expected_drift_locs=expected_drift_locs,
                acc_per_drift_logger=acc_per_drift_logger,
                stream_sequences=stream_sequences)

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

        else:
            pearl = pearl(args.num_trees,
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
                          args.enable_state_adaption,
                          args.enable_state_graph)
            evaluator = Evaluator()
            evaluator.prequential_evaluation(
                    classifier=pearl,
                    stream=data_file_path,
                    max_samples=args.max_samples,
                    sample_freq=args.sample_freq,
                    metrics_logger=metrics_logger,
                    expected_drift_locs=expected_drift_locs,
                    acc_per_drift_logger=acc_per_drift_logger)
