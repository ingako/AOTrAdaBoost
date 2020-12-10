import copy
from collections import deque
import math
from random import randrange
import time

import numpy as np
from sklearn.metrics import cohen_kappa_score

import sys
paths = [r'..', r'../third_party']

for path in paths:
    if path not in sys.path:
        sys.path.append(path)

# from build.trans_pearl import pearl, trans_pearl
from trans_pearl import pearl, trans_pearl
class Evaluator:

    # def prequential_evaluation_transfer(classifier,
    #                                     stream,
    #                                     max_samples,
    #                                     sample_freq,
    #                                     metrics_logger,
    #                                     expected_drift_locs,
    #                                     acc_per_drift_logger):


    @staticmethod
    def prequential_evaluation_transfer(
                    classifier,
                    stream,
                    max_samples,
                    sample_freq,
                    metrics_logger,
                    seq_logger,
                    pro_drift_window,
                    drift_interval_seq_len,
                    expected_drift_locs,
                                        acc_per_drift_logger):

        # 1. Match a similar source concept
        # 2. Generate pseudo data
        # 3. Tradaboost to target concept (decided #instances later)

        correct = 0
        acc_per_drift_correct = 0
        window_actual_labels = []
        window_predicted_labels = []

        start_time = time.process_time()

        classifier.init_data_source(stream);

        # for count in range(0, max_samples):
        for count in range(0, 10000):
            if not classifier.get_next_instance():
                break

            # test
            prediction = classifier.predict()

            actual_label = classifier.get_cur_instance_label()
            if prediction == actual_label:
                correct += 1

                if expected_drift_locs:
                    if count > expected_drift_locs[0] + 1000:
                        expected_drift_locs.popleft()
                        acc_per_drift_logger.info(acc_per_drift_correct/1000)
                        acc_per_drift_correct = 0
                    if len(expected_drift_locs) > 0 \
                            and count > expected_drift_locs[0] \
                            and count < expected_drift_locs[0] + 1000:
                        acc_per_drift_correct += 1

            window_actual_labels.append(actual_label)
            window_predicted_labels.append(prediction)

            # train
            classifier.train()

            classifier.delete_cur_instance()

            if count % sample_freq == 0 and count != 0:
                elapsed_time = time.process_time() - start_time
                accuracy = correct / sample_freq
                kappa = cohen_kappa_score(window_actual_labels, window_predicted_labels)

                candidate_tree_size = 0
                tree_pool_size = 60

                candidate_tree_size = classifier.get_candidate_tree_group_size()
                tree_pool_size = classifier.get_tree_pool_size()

                print(f"{count},{accuracy},{kappa},{candidate_tree_size},{tree_pool_size},{elapsed_time}")
                metrics_logger.info(f"{count},{accuracy},{kappa}," \
                                    f"{candidate_tree_size},{tree_pool_size},{elapsed_time}")

                correct = 0
                window_actual_labels = []
                window_predicted_labels = []

        # classifier.init_data_source(stream);
        classifier.generate_data(0, 1)
