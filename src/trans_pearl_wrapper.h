#ifndef __TRANS_PEARL_WRAPPER_H__
#define __TRANS_PEARL_WRAPPER_H__

#include "trans_pearl.h"

class trans_pearl_wrapper {
public:

    trans_pearl_wrapper(int num_classifiers,
                        int num_trees,
                        int max_num_candidate_trees,
                        int repo_size,
                        int edit_distance_threshold,
                        int kappa_window_size,
                        int lossy_window_size,
                        int reuse_window_size,
                        int arf_max_features,
                        int lambda,
                        int seed,
                        double bg_kappa_threshold,
                        double cd_kappa_threshold,
                        double reuse_rate_upper_bound,
                        double warning_delta,
                        double drift_delta,
                        int least_transfer_warning_period_instances_length,
                        int instance_store_size,
                        int num_pseudo_instances,
                        int bbt_pool_size,
                        int eviction_interval,
                        double transfer_kappa_threshold,
                        string boost_mode_str);

    trans_pearl_wrapper(int num_classifiers,
                        int num_trees,
                        int max_num_candidate_trees,
                        int repo_size,
                        int edit_distance_threshold,
                        int kappa_window_size,
                        int lossy_window_size,
                        int reuse_window_size,
                        int arf_max_features,
                        int lambda,
                        int seed,
                        double bg_kappa_threshold,
                        double cd_kappa_threshold,
                        double reuse_rate_upper_bound,
                        double warning_delta,
                        double drift_delta);

    void switch_classifier(int classifier_idx);
    void train();
    int predict();
    int get_cur_instance_label();
    void init_data_source(int classifier_idx, const string &filename);
    bool get_next_instance();
    int get_candidate_tree_group_size();
    int get_transferred_tree_group_size();
    int get_tree_pool_size();

private:

    vector<shared_ptr<pearl>> classifiers;
    shared_ptr<pearl> current_classifier;

};

#endif