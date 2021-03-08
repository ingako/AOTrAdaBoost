#include "trans_pearl_wrapper.h"
#include "trans_pearl.h"

trans_pearl_wrapper::trans_pearl_wrapper(
                    int num_classifiers,
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
                    int num_diff_distr_instances,
                    int bbt_pool_size,
                    int eviction_interval,
                    double transfer_kappa_threshold,
                    string boost_mode_str) {

    for (int i = 0; i < num_classifiers; i++) {
        shared_ptr<trans_pearl> classifier = make_shared<trans_pearl>(
                num_trees,
                max_num_candidate_trees,
                repo_size,
                edit_distance_threshold,
                kappa_window_size,
                lossy_window_size,
                reuse_window_size,
                arf_max_features,
                lambda,
                seed,
                bg_kappa_threshold,
                cd_kappa_threshold,
                reuse_rate_upper_bound,
                warning_delta,
                drift_delta,
                least_transfer_warning_period_instances_length,
                instance_store_size,
                num_diff_distr_instances,
                bbt_pool_size,
                eviction_interval,
                transfer_kappa_threshold,
                boost_mode_str);

        classifiers.push_back(classifier);
    }

    for (int i = 0; i < num_classifiers; i++) {
        for (int j = 0; j < num_classifiers; j++) {
            if (i == j) continue;
            shared_ptr<trans_pearl> from_trans_pearl_classifier = static_pointer_cast<trans_pearl>(classifiers[i]);
            shared_ptr<trans_pearl> to_trans_pearl_classifier = static_pointer_cast<trans_pearl>(classifiers[j]);
            from_trans_pearl_classifier->register_tree_pool(to_trans_pearl_classifier->get_concept_repo());
        }
    }
}

trans_pearl_wrapper::trans_pearl_wrapper(
        int num_classifiers,
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
        double drift_delta) {

    for (int i = 0; i < num_classifiers; i++) {
        shared_ptr<pearl> classifier = make_shared<pearl>(
                num_trees,
                max_num_candidate_trees,
                repo_size,
                edit_distance_threshold,
                kappa_window_size,
                lossy_window_size,
                reuse_window_size,
                arf_max_features,
                lambda,
                seed,
                bg_kappa_threshold,
                cd_kappa_threshold,
                reuse_rate_upper_bound,
                warning_delta,
                drift_delta,
                true,
                true);

        classifiers.push_back(classifier);
    }
}

void trans_pearl_wrapper::switch_classifier(int classifier_idx){
    current_classifier = classifiers[classifier_idx];
}

void trans_pearl_wrapper::train() {
    current_classifier->train();
}

int trans_pearl_wrapper::predict() {
    return current_classifier->predict();
}

int trans_pearl_wrapper::get_cur_instance_label() {
    return current_classifier->get_cur_instance_label();
}

void trans_pearl_wrapper::init_data_source(int classifier_idx, const string &filename){
    classifiers[classifier_idx]->init_data_source(filename);
}

bool trans_pearl_wrapper::get_next_instance() {
    return current_classifier->get_next_instance();
}

int trans_pearl_wrapper::get_candidate_tree_group_size() {
    return current_classifier->get_candidate_tree_group_size();
}

int trans_pearl_wrapper::get_transferred_tree_group_size() {
    shared_ptr<trans_pearl> trans_pearl_classifier = static_pointer_cast<trans_pearl>(current_classifier);
    return trans_pearl_classifier->get_transferred_tree_group_size();
}

int trans_pearl_wrapper::get_tree_pool_size() {
    return current_classifier->get_tree_pool_size();
}
