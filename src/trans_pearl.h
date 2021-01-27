#ifndef __TRANS_PEARL_H__
#define __TRANS_PEARL_H__

#include "PEARL/src/cpp/pearl.h"
#include "knn-cpp/include/knn/kdtree_minkowski.h"

typedef Eigen::MatrixXd Matrix;
typedef knn::Matrixi Matrixi;

class trans_pearl_tree;

class trans_pearl : public pearl {
    class boosted_bg_tree_pool;

    public:

        trans_pearl(int num_trees,
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
                  int pro_drift_window_size,
                  double hybrid_delta,
                  int backtrack_window,
                  double stability_delta);

        virtual void train();
        virtual shared_ptr<pearl_tree> make_pearl_tree(int tree_pool_id);
        virtual void init();

        int find_last_actual_drift_point(int tree_idx);
        void set_expected_drift_prob(int tree_idx, double p);
        bool has_actual_drift(int tree_idx);
        void update_drifted_tree_indices(const vector<int>& tree_indices);
        vector<int> get_stable_tree_indices();

        void select_predicted_trees(const vector<int>& warning_tree_pos_list);

        vector<int> adapt_state(const vector<int>& drifted_tree_pos_list, bool is_proactive);
        vector<int> adapt_state_with_proactivity(
                const vector<int>& drifted_tree_pos_list,
                deque<shared_ptr<pearl_tree>>& _candidate_trees);

        bool has_actual_drifted_trees();
        void evaluate_tree(vector<Instance*> &pseudo_instances);
        vector<Instance*> generate_data(int tree_idx, int num_instances);
        void transfer();

    private:

        int pro_drift_window_size = 100;
        double hybrid_delta = 0.001;
        int backtrack_window = 25;
        double stability_delta = 0.001;

        int num_max_backtrack_instances = 100000000; // TODO
        int num_instances_seen = 0;
        deque<Instance*> backtrack_instances;
        set<int> potential_drifted_tree_indices;
        vector<unique_ptr<HT::ADWIN>> stability_detectors;
        vector<int> stable_tree_indices;
        deque<shared_ptr<pearl_tree>> predicted_trees;

        static bool compare_kappa_arf(shared_ptr<arf_tree>& tree1,
                                          shared_ptr<arf_tree>& tree2);
        // virtual void predict_with_state_adaption(vector<int>& votes, int actual_label);
        bool detect_stability(int error_count, unique_ptr<HT::ADWIN>& detector);

        // Transfer
        vector<int> actual_drifted_trees;
        vector<double> best_perf_metrics_for_drifted_trees;
        vector<vector<Instance*>> instance_stores;
        vector<DenseInstance*> find_k_closest_instances(DenseInstance* target_instance,
                                                            vector<Instance*>& instance_store,
                                                            int k);

        // ozaboost
        vector<double> scms;
        vector<double> swms;
        long training_weights_seen_by_model = 0;
        double getEnsembleMemberWeight(int tree_idx);
        virtual int predict();


        // boosting for transfer learning
        int stream_instance_idx = 0;
        vector<int> drift_warning_period_lengths;

        int pool_size = 10;
        int mini_batch_size = 100;
        // one boosted background tree pool per foreground tree
        vector<unique_ptr<boosted_bg_tree_pool>> bbt_pools;

        class boosted_bg_tree_pool {
        public:
            boosted_bg_tree_pool(int pool_size,
                                 shared_ptr<trans_pearl_tree> tree_template);

            // training starts when a mini_batch is ready
            void train(Instance* instance, bool is_same_distribution);
            shared_ptr<trans_pearl_tree> get_best_model();

        private:
            long pool_size = 10;
            long bbt_counter = 0;
            shared_ptr<trans_pearl_tree> tree_template;
            vector<Instance*> mini_batch;
            vector<double> instance_weights;
            vector<double> model_weights;
            vector<shared_ptr<trans_pearl_tree>> pool;
            vector<double> oob_errors_vec; // out-of-bag errors per boosted bg tree

            // data comes from the same distribution during drift warning period
            bool is_same_distribution = true;

            // execute replacement strategies when the bbt pool is full
            void update_bbt();
            void boost(int is_same_distribution);
        };

};

class trans_pearl_tree : public pearl_tree {
public:
    trans_pearl_tree(int tree_pool_id,
                     int kappa_window_size,
                     int pro_drift_window_size,
                     double warning_delta,
                     double drift_delta,
                     double hybrid_delta,
                     std::mt19937 mrand);

    trans_pearl_tree(trans_pearl_tree const &rhs);

    vector<Instance*> instance_store;

    virtual void train(Instance &instance);
};

#endif