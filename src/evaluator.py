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
# from trans_pearl import pearl, trans_pearl

class ClassifierMetrics:
    def __init__(self):
        self.correct = 0
        self.acc_per_drift_correct = 0
        self.window_actual_labels = []
        self.window_predicted_labels = []
        self.start_time = 0
        self.total_time = 0
        self.instance_idx = 0


class Evaluator:

    def prequential_evaluation_transfer(
                    self,
                    classifier,
                    data_file_paths,
                    max_samples,
                    sample_freq,
                    metrics_loggers,
                    seq_logger,
                    expected_drift_locs,
                    acc_per_drift_logger,
                    stream_sequences):

        classifier_metrics_list = []
        for i in range(len(data_file_paths)):
            classifier.init_data_source(i, data_file_paths[i])
            classifier_metrics_list.append(ClassifierMetrics())

        stream_sequence = stream_sequences.popleft()
        classifier_idx, switch_location = stream_sequence[0], stream_sequence[1]
        classifier.switch_classifier(classifier_idx)
        metric = classifier_metrics_list[classifier_idx]
        classifier_metrics_list[classifier_idx].start_time = time.process_time()

        # for count in range(0, max_samples):
        for count in range(0, 61000):
            # TODO
            if count == switch_location and len(stream_sequences) > 0:
                # Switch streams to simulate parallel streams
                metric.total_time += time.process_time() - metric.start_time

                stream_sequence = stream_sequences.popleft()
                classifier_idx, switch_location = stream_sequence[0], stream_sequence[1]
                classifier.switch_classifier(classifier_idx)
                metric = classifier_metrics_list[classifier_idx]
                metric.start_time = time.process_time()

                print()
                print(f"switching to classifier_idx {classifier_idx}")

            if not classifier.get_next_instance():
                break
            classifier_metrics_list[classifier_idx].instance_idx += 1

            # test
            prediction = classifier.predict()

            actual_label = classifier.get_cur_instance_label()
            if prediction == actual_label:
                metric.correct += 1

                if expected_drift_locs:
                    if count > expected_drift_locs[0] + 1000:
                        expected_drift_locs.popleft()
                        acc_per_drift_logger.info(metric.acc_per_drift_correct/1000)
                        metric.acc_per_drift_correct = 0
                    if len(expected_drift_locs) > 0 \
                            and expected_drift_locs[0] < count < expected_drift_locs[0] + 1000:
                        metric.acc_per_drift_correct += 1

            metric.window_actual_labels.append(actual_label)
            metric.window_predicted_labels.append(prediction)

            # train
            classifier.train()
            # if classifier.has_actual_drifted_trees():
            #     self._transfer(classifier)

            # classifier.delete_cur_instance()
            self._log_metrics(classifier_metrics_list[classifier_idx].instance_idx, sample_freq, metric, classifier, metrics_loggers[classifier_idx])

    def _log_metrics(self, count, sample_freq, metric, classifier, metrics_logger):
        if count % sample_freq == 0 and count != 0:
            elapsed_time = time.process_time() - metric.start_time
            accuracy = metric.correct / sample_freq
            kappa = cohen_kappa_score(metric.window_actual_labels,
                                      metric.window_predicted_labels)

            candidate_tree_size = classifier.get_candidate_tree_group_size()
            transferred_tree_size = classifier.get_transferred_tree_group_size()
            tree_pool_size = classifier.get_tree_pool_size()

            # TODO multiple output streams
            print(f"{count},{accuracy},{kappa},{candidate_tree_size},{transferred_tree_size},{tree_pool_size},{elapsed_time}")
            metrics_logger.info(f"{count},{accuracy},{kappa},"
                                f"{candidate_tree_size},{transferred_tree_size},{tree_pool_size},{elapsed_time}")
            if transferred_tree_size > 0:
                print(f"----------------transferred {transferred_tree_size} into foreground")

            metric.correct = 0
            metric.window_actual_labels = []
            metric.window_predicted_labels = []
