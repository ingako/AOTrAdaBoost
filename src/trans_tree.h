#ifndef TRANS_TREE_H
#define TRANS_TREE_H

#include <streamDM/streams/ArffReader.h>
#include <streamDM/learners/Classifiers/Trees/HoeffdingTree.h>
#include <streamDM/learners/Classifiers/Trees/ADWIN.h>
#include "trans_tree_wrapper.h"

enum boost_modes { no_boost_mode, ozaboost_mode, tradaboost_mode, otradaboost_mode };

class hoeffding_tree {
public:
    hoeffding_tree(double warning_delta, double drift_delta);
    void train(Instance& instance);
    int predict(Instance& instance);
    unique_ptr<HT::HoeffdingTree> tree;
    shared_ptr<hoeffding_tree> bg_tree;
    unique_ptr<HT::ADWIN> warning_detector;
    unique_ptr<HT::ADWIN> drift_detector;

private:
    double warning_delta;
    double drift_delta;
};


class trans_tree {
    class boosted_bg_tree_pool;

public:
    trans_tree(
            int seed,
            int kappa_window_size,
            double warning_delta,
            double drift_delta,
            // transfer learning params
            int least_transfer_warning_period_instances_length, // tuning required
            int instance_store_size,
            int num_diff_distr_instances,
            int bbt_pool_size, // tuning required
            int eviction_interval,
            double transfer_kappa_threshold,
            string boost_mode_str);

    void train();
    int predict();
    void init();
    shared_ptr<hoeffding_tree> make_tree(int tree_pool_id);
    bool detect_change(int error_count, unique_ptr<HT::ADWIN>& detector);

    bool init_data_source(const string& filename);
    bool get_next_instance();
    int get_cur_instance_label();
    void delete_cur_instance();

    // transfer
    vector<shared_ptr<hoeffding_tree>>& get_concept_repo();
    void register_tree_pool(vector<shared_ptr<hoeffding_tree>>& pool);
    shared_ptr<hoeffding_tree> match_concept(vector<Instance*> warning_period_instances);
    int get_transferred_tree_group_size() const;

private:

    int kappa_window_size;
    double warning_delta;
    double drift_delta;
    std::mt19937 mrand;
    shared_ptr<hoeffding_tree> foreground_tree;
    vector<shared_ptr<hoeffding_tree>> tree_pool;
    deque<int> actual_labels;

    Instance* instance;
    unique_ptr<Reader> reader;

    // transfer
    std::map<string, boost_modes> boost_mode_map =
            {
                    { "no_boost", no_boost_mode},
                    { "ozaboost", ozaboost_mode },
                    { "tradaboost", tradaboost_mode },
                    { "otradaboost", otradaboost_mode },
            };
    boost_modes boost_mode = otradaboost_mode;
    int least_transfer_warning_period_length = 50;
    int instance_store_size = 500;
    int num_diff_distr_instances = 30;
    int bbt_pool_size = 100;
    int eviction_interval = 100;
    double transfer_kappa_threshold = 0.3;
    unique_ptr<boosted_bg_tree_pool> bbt_pool;

    class boosted_bg_tree_pool {
    public:
        boost_modes boost_mode = otradaboost_mode;

        boosted_bg_tree_pool(enum boost_modes boost_mode,
                             int pool_size,
                             int eviction_interval,
                             double transfer_kappa_threshold,
                             shared_ptr<hoeffding_tree> tree_template,
                             int lambda);

        // training starts when a mini_batch is ready
        void train(Instance* instance, bool is_same_distribution);
        shared_ptr<hoeffding_tree> get_best_model(deque<int> actual_labels, int class_count);
        void online_boost(Instance* instance, bool _is_same_distribution);
        Instance* get_next_diff_distr_instance();

        vector<Instance*> warning_period_instances;
        shared_ptr<hoeffding_tree> matched_tree = nullptr;
        int instance_store_idx = 0;

    private:
        double lambda = 1;
        double epsilon = 1;
        std::mt19937 mrand;

        long pool_size = 10;
        long bbt_counter = 0;
        long boost_count = 0;
        long eviction_interval = 100;
        double transfer_kappa_threshold = 0.3;
        shared_ptr<hoeffding_tree> tree_template;
        vector<shared_ptr<hoeffding_tree>> pool;
        vector<double> oob_tree_correct_lam_sum; // count of out-of-bag correctly predicted trees per instance
        vector<double> oob_tree_wrong_lam_sum; // count of out-of-bag incorrectly predicted trees per instance
        vector<double> oob_tree_lam_sum; // count of oob trees per instance

        // execute replacement strategies when the bbt pool is full
        void update_bbt();
        void no_boost(Instance* instance);
        void ozaboost(Instance* instance);
        void tradaboost(Instance* instance, bool is_same_distribution);
        void otradaboost(Instance* instance, bool is_same_distribution);
        void perf_eval(Instance* instance);
    };


};

#endif //TRANS_TREE_H
