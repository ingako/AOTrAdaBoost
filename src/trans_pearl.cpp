#include "trans_pearl.h"

trans_pearl::trans_pearl(int num_trees,
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
                         // transfer learning params
                         int least_transfer_warning_period_instances_length,
                         int instance_store_size,
                         int num_pseudo_instances,
                         int bbt_pool_size,
                         int mini_batch_size):
        pearl(num_trees,
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
              true),
        least_transfer_warning_period_length(least_transfer_warning_period_instances_length), // 50 pro_drift_window_size(pro_drift_window_size),
        instance_store_size(instance_store_size),
        num_pseudo_instances(num_pseudo_instances),
        bbt_pool_size(bbt_pool_size),
        mini_batch_size(mini_batch_size) {

}

void trans_pearl::init() {
    vector<shared_ptr<trans_pearl_tree>> temp_tree_pool = vector<shared_ptr<trans_pearl_tree>>(num_trees);
    stability_detectors = vector<unique_ptr<HT::ADWIN>>();

    for (int i = 0; i < num_trees; i++) {
        temp_tree_pool[i] = static_pointer_cast<trans_pearl_tree>( make_pearl_tree(i));
        foreground_trees.push_back(temp_tree_pool[i]);
        stability_detectors.push_back(make_unique<HT::ADWIN>(warning_delta));
        bbt_pools.push_back(nullptr);
    }

    for (auto t : temp_tree_pool) {
        tree_pool.push_back(t);
    }

    // scms.resize(num_trees, 0);
    // swms.resize(num_trees, 0);
}

shared_ptr<pearl_tree> trans_pearl::make_pearl_tree(int tree_pool_id) {
    return make_shared<trans_pearl_tree>(tree_pool_id,
                                         kappa_window_size,
                                         warning_delta,
                                         drift_delta,
                                         mrand,
                                         instance_store_size);
}

// foreground trees make predictions, update votes, keep track of actual labels
// find warning trees, select candidate trees
// find drifted trees, update potential_drifted_tree_indices
// candidate trees make predictions
void trans_pearl::train() {
    stream_instance_idx += 1;
    actual_drifted_trees.clear();

    if (drift_warning_period_lengths.size() == 0) {
        drift_warning_period_lengths.resize(num_trees, 0);
    }
    if (stream_instance_idx == 25000) {
        cout << "drift_warning_distance: " << endl;
        for (int i = 0; i < num_trees; i++) {
            cout << drift_warning_period_lengths[i] << " ";
        }
        cout << endl;
    }


    if (foreground_trees.empty()) {
        init();
    }

    potential_drifted_tree_indices.clear();
    stable_tree_indices.clear();

    backtrack_instances.push_back(instance);
    num_instances_seen++;

    int actual_label = instance->getLabel();

    // keep track of actual labels for candidate tree evaluations
    if (actual_labels.size() >= kappa_window_size) {
        actual_labels.pop_front();
    }
    actual_labels.push_back(actual_label);

    vector<int> warning_tree_pos_list;
    vector<int> drifted_tree_pos_list;

    shared_ptr<pearl_tree> cur_tree = nullptr;


    // ozaboost
    // double lambda_d = this->lambda;
    // training_weights_seen_by_model += 1; // TODO

    for (int i = 0; i < num_trees; i++) {
        if (drift_warning_period_lengths[i] > 0) {
            drift_warning_period_lengths[i]++;
        }

        // online bagging
        std::poisson_distribution<int> poisson_distr(lambda);
        int weight = poisson_distr(mrand);

        // ozaboost
        // std::poisson_distribution<int> poisson_distr(lambda_d);
        // int weight = poisson_distr(mrand);

        if (weight == 0) {
            continue;
        }

        instance->setWeight(weight);

        cur_tree = static_pointer_cast<pearl_tree>(foreground_trees[i]);
        cur_tree->train(*instance);

        int predicted_label = cur_tree->predict(*instance, true);
        int error_count = (int) (predicted_label != actual_label);

        bool warning_detected_only = false;

        // detect warning
        if (detect_change(error_count, cur_tree->warning_detector)) {
            warning_detected_only = true;
            cur_tree->bg_pearl_tree = make_pearl_tree(-1);
            cur_tree->warning_detector->resetChange();
            if (drift_warning_period_lengths[i] == 0) {
                drift_warning_period_lengths[i] = 1;
            }

            shared_ptr<trans_pearl_tree> tree_template
                    = static_pointer_cast<trans_pearl_tree>(cur_tree->bg_pearl_tree);
            bbt_pools[i] = make_unique<boosted_bg_tree_pool>(bbt_pool_size, mini_batch_size, tree_template, this->lambda);
        }

        // detect drift
        if (detect_change(error_count, cur_tree->drift_detector)) {
            warning_detected_only = false;
            drifted_tree_pos_list.push_back(i);
            potential_drifted_tree_indices.insert(i);

            cur_tree->warning_detector->resetChange();
            cur_tree->drift_detector->resetChange();

            if (drift_warning_period_lengths[i] > 0) {
                drift_warning_period_lengths[i] = -drift_warning_period_lengths[i];
            }
        }

        if (warning_detected_only) {
            warning_tree_pos_list.push_back(i);
        }

        // detect stability
        // int correct_count = (int) (actual_label == predicted_label);
        // if (cur_tree->replaced_tree
        //     && detect_stability(correct_count, stability_detectors[i])) {
        //     stability_detectors[i] = make_unique<HT::ADWIN>(stability_delta);
        //     stable_tree_indices.push_back(i);
        // }

        if (bbt_pools[i] != nullptr) {
            // Add instance to mini-batch. Boosting auto triggers when the mini-batch is full.
            bbt_pools[i]->online_tradaboost(instance->clone(), true, false);
            bbt_pools[i]->warning_period_instances.push_back(instance->clone());
        }

        // ozaboost: update weights
        // if (error_count == 0) {
        //     this->scms[i] += lambda_d;
        //     lambda_d *= this->training_weights_seen_by_model / (2 * this->scms[i]);

        // } else {
        //     this->swms[i] += lambda_d;
        //     lambda_d *= this->training_weights_seen_by_model / (2 * this->swms[i]);
        // }
    }

    for (int i = 0; i < candidate_trees.size(); i++) {
        candidate_trees[i]->predict(*instance, true);
    }

    for (int i = 0; i < predicted_trees.size(); i++) {
        predicted_trees[i]->predict(*instance, true);
    }

    // if warnings are detected, find closest state and update candidate_trees list
    if (warning_tree_pos_list.size() > 0) {
        pearl::select_candidate_trees(warning_tree_pos_list);
    }

    // if actual drifts are detected, swap trees and update cur_state
    if (drifted_tree_pos_list.size() > 0) {
        transferred_foreground_pos_list.clear();
        transfer(drifted_tree_pos_list);

        actual_drifted_trees = adapt_state(drifted_tree_pos_list, candidate_trees, false);

        transferred_tree_total_count += transferred_foreground_pos_list.size();
    }
}

void trans_pearl::select_predicted_trees(const vector<int>& warning_tree_pos_list) {

    if (enable_state_graph) {
        // try trigger lossy counting
        if (state_graph->update(warning_tree_pos_list.size())) {
            // TODO log
        }
    }

    // add selected neighbors as candidate trees if graph is stable
    if (state_graph->get_is_stable()) {
        tree_transition(warning_tree_pos_list, predicted_trees);
    }

    // trigger pattern matching if graph has become unstable
    if (!state_graph->get_is_stable()) {
        pattern_match_candidate_trees(warning_tree_pos_list, predicted_trees);

    } else {
        // TODO log
    }
}

bool trans_pearl::has_actual_drift(int tree_idx) {
    if (potential_drifted_tree_indices.find(tree_idx) != potential_drifted_tree_indices.end()) {
        return false;

    }

    shared_ptr<pearl_tree> cur_tree
        = static_pointer_cast<pearl_tree>(foreground_trees[tree_idx]);

    return cur_tree->has_actual_drift();
}

void trans_pearl::update_drifted_tree_indices(const vector<int>& tree_indices) {
    // for (int idx: tree_indices) {
    //     if(potential_drifted_tree_indices.find(element) == potential_drifted_tree_indices.end()) {
    //         actual_drifted_predicted_tree_indices.insert(idx);
    //     }
    // }
}

vector<int> trans_pearl::adapt_state(
        const vector<int>& drifted_tree_pos_list,
        deque<shared_ptr<pearl_tree>>& _candidate_trees,
        bool is_transferred_tree) {

    vector<int> actual_drifted_tree_indices; // for return

    int class_count = instance->getNumberClasses();

    // sort candidate trees by kappa
    for (int i = 0; i < _candidate_trees.size(); i++) {
        _candidate_trees[i]->update_kappa(actual_labels, class_count);
    }
    sort(_candidate_trees.begin(), _candidate_trees.end(), compare_kappa);

    if (is_transferred_tree) {
        cout << "kappa: ";
        for (int i = 0; i < _candidate_trees.size(); i++) {
            cout << _candidate_trees[i]->kappa << " ";
            if (_candidate_trees[i]->kappa < 0.1) {
                _candidate_trees.pop_front();
            }
        }
        cout << endl;
    }

    for (int i = 0; i < drifted_tree_pos_list.size(); i++) {
        // TODO
        if (tree_pool.size() >= repo_size) {
            std::cout << "tree_pool full: "
                      << std::to_string(tree_pool.size()) << endl;
            exit(1);
        }

        int drifted_pos = drifted_tree_pos_list[i];
        shared_ptr<pearl_tree> drifted_tree = static_pointer_cast<pearl_tree>(foreground_trees[drifted_pos]);
        shared_ptr<pearl_tree> swap_tree = nullptr;

        drifted_tree->update_kappa(actual_labels, class_count);
        // if (drifted_tree->kappa == INT_MIN) {
        //     continue;
        // }

        cur_state.erase(drifted_tree->tree_pool_id);

        bool add_to_repo = false;

        if (_candidate_trees.size() > 0
                && _candidate_trees.back()->kappa
                    - drifted_tree->kappa >= cd_kappa_threshold) {

            actual_drifted_tree_indices.push_back(drifted_tree_pos_list[i]);

            _candidate_trees.back()->is_candidate = false;
            swap_tree = _candidate_trees.back();
            _candidate_trees.pop_back();

            if (enable_state_graph) {
                graph_switch->update_reuse_count(1);
            }

        }

        if (swap_tree == nullptr && !is_transferred_tree) {
            add_to_repo = true;

            if (enable_state_graph) {
                graph_switch->update_reuse_count(0);
            }

            shared_ptr<pearl_tree> bg_tree = drifted_tree->bg_pearl_tree;

            if (!bg_tree) {
                swap_tree = make_pearl_tree(tree_pool.size());

            } else {
                bg_tree->update_kappa(actual_labels, class_count);

                if (bg_tree->kappa == INT_MIN) {
                    // add bg tree to the repo even if it didn't fill the window

                } else if (bg_tree->kappa - drifted_tree->kappa >= bg_kappa_threshold) {

                } else {
                    // false positive
                    add_to_repo = false;

                }

                swap_tree = bg_tree;
            }

            if (add_to_repo) {
                swap_tree->reset();

                // assign a new tree_pool_id for background tree
                // and allocate a slot for background tree in tree_pool
                swap_tree->tree_pool_id = tree_pool.size();
                tree_pool.push_back(swap_tree);

                actual_drifted_tree_indices.push_back(drifted_tree_pos_list[i]);

            } else {
                // false positive
                swap_tree->tree_pool_id = drifted_tree->tree_pool_id;

                // TODO
                // swap_tree = move(drifted_tree);
            }
        }

        if (!swap_tree) {
            if (is_transferred_tree) {
                continue;
            }
            LOG("adapt state with transferred trees: swap_tree is nullptr");
            exit(1);
        }

        // log the number of transferred trees that swapped drifted trees
        if (is_transferred_tree) {
            cout << "swapping with transfered_tree on " << drifted_pos << "------------------" << endl;
            cout << swap_tree->kappa << endl;
            transferred_foreground_pos_list.push_back(drifted_pos);
        } else {
            auto position = std::find(transferred_foreground_pos_list.begin(),
                                      transferred_foreground_pos_list.end(),
                                      drifted_pos);
            if (position != transferred_foreground_pos_list.end()) {
                transferred_foreground_pos_list.erase(position);
                cout << "erased transfered_tree on " << drifted_pos << "------------------" << endl;
            }
        }

        // start swapping trees

        if (enable_state_graph) {
            if (swap_tree->tree_pool_id == -1) {
                if (!is_transferred_tree) {
                    LOG("non transferred swap tree does not have a tree_pool_id");
                    exit(1);
                }
                swap_tree->tree_pool_id = tree_pool.size();
                tree_pool.push_back(swap_tree);
            }
            state_graph->add_edge(drifted_tree->tree_pool_id, swap_tree->tree_pool_id);
        }

        cur_state.insert(swap_tree->tree_pool_id);

        // keep track of drifted tree
        swap_tree->replaced_tree = drifted_tree;

        // replace drifted_tree with swap tree
        foreground_trees[drifted_pos] = swap_tree;

        drifted_tree->reset();
    }

    state_queue->enqueue(cur_state);

    if (enable_state_graph) {
        graph_switch->update_switch();
    }

    return actual_drifted_tree_indices;
}

int trans_pearl::get_transferred_tree_group_size() const {
    return this->transferred_tree_total_count;
}

void trans_pearl::set_expected_drift_prob(int tree_idx, double p) {
    shared_ptr<pearl_tree> cur_tree = nullptr;
    cur_tree = static_pointer_cast<pearl_tree>(foreground_trees[tree_idx]);
    cur_tree->set_expected_drift_prob(p);
}

bool trans_pearl::compare_kappa_arf(shared_ptr<arf_tree>& tree1,
                                  shared_ptr<arf_tree>& tree2) {
    shared_ptr<pearl_tree> pearl_tree1 =
        static_pointer_cast<pearl_tree>(tree1);
    shared_ptr<pearl_tree> pearl_tree2 =
        static_pointer_cast<pearl_tree>(tree2);

    return pearl_tree1->kappa < pearl_tree2->kappa;
}

bool trans_pearl::detect_stability(int error_count,
                                 unique_ptr<HT::ADWIN>& detector) {

    double old_error = detector->getEstimation();
    bool error_change = detector->setInput(error_count);

    if (!error_change) {
       return false;
    }

    if (old_error > detector->getEstimation()) {
        // error is decreasing
        // cout << "error is decreasing" << endl;
        return true;
    }

    return false;
}

vector<int> trans_pearl::get_stable_tree_indices() {
    return stable_tree_indices;
}

bool trans_pearl::has_actual_drifted_trees() {
    return actual_drifted_trees.size() > 0;
}

void trans_pearl::transfer(vector<int>& actual_drifted_trees) {
    // For each actual drifted trees
    // 1. Concept Matching
    // 2. Boosting
    // 3. Evaluate on warning period data
    // 4. Replace candidate trees (which replaced actual drifted trees)

    deque<shared_ptr<pearl_tree>> transfer_trees;

    for (auto drifted_tree_idx : actual_drifted_trees) {
        if (bbt_pools[drifted_tree_idx] == nullptr) {
            // cout << "transfer: bbt_pool does not exist for an actual drifted tree"  << endl;
            // exit(1);
            return;
        }

        if (bbt_pools[drifted_tree_idx]->warning_period_instances.size() < least_transfer_warning_period_length) {
            // cout << "warning_period_instances size is not enough" << endl;
            return;
        }

        shared_ptr<trans_pearl_tree> drifted_trans_pearl_tree =
                static_pointer_cast<trans_pearl_tree>(foreground_trees[drifted_tree_idx]);
        shared_ptr<trans_pearl_tree> bg_trans_pearl_tree =
                static_pointer_cast<trans_pearl_tree>(drifted_trans_pearl_tree->bg_pearl_tree);
        if (bg_trans_pearl_tree== nullptr) {
            // cout << "transfer: bg_drifted_tree is nullptr" << endl;
            return;
        }


        shared_ptr<trans_pearl_tree> matched_tree = match_concept(bg_trans_pearl_tree);
        if (matched_tree == nullptr) {
            continue;
        }

        // TODO stopping criteria
        // cout << "generating pseudo_instances" << endl;
        vector<Instance*> pseudo_instances = matched_tree->generate_data(instance, num_pseudo_instances);
        bbt_pools[drifted_tree_idx]->is_same_distribution = false;
        for (auto pseudo_instance : pseudo_instances) {
            bbt_pools[drifted_tree_idx]->online_tradaboost(pseudo_instance, false, false);
        }

        vector<shared_ptr<pearl_tree>> best_models = bbt_pools[drifted_tree_idx]->get_best_models();
        transfer_trees.insert(std::end(transfer_trees), std::begin(best_models), std::end(best_models));

        bbt_pools[drifted_tree_idx] = nullptr;
    }

    if (transfer_trees.size() > 0) {
        cout << "adapting state - size of transfer_trees: " << transfer_trees.size() << endl;
        adapt_state(actual_drifted_trees, transfer_trees, true);
    }
}

shared_ptr<trans_pearl_tree> trans_pearl::match_concept(shared_ptr<trans_pearl_tree> drifted_tree) {
    if (drifted_tree->instance_store.size() < least_transfer_warning_period_length) {
        cout << "match_concept: drifted_tree does not have enough warning_period_instance" << endl;
        return nullptr;
    }

    shared_ptr<trans_pearl_tree> matched_tree = nullptr;
    double highest_kappa = 0.0;

    // For kappa calculation
    int class_count = drifted_tree->instance_store[0]->getNumberClasses();
    vector<int> true_labels;
    for (auto warning_period_instance : drifted_tree->instance_store) {
        true_labels.push_back(warning_period_instance->getLabel());
    }

    for (auto registered_tree_pool : registered_tree_pools) {
        for (auto tree : *registered_tree_pool) {
            shared_ptr<trans_pearl_tree> trans_tree = static_pointer_cast<trans_pearl_tree>(tree);

            vector<int> predicted_labels;
            for (auto warning_period_instance : drifted_tree->instance_store) {
                // TODO: reset performance tracking?
                int prediction = trans_tree->predict(*warning_period_instance, false);
                predicted_labels.push_back(prediction);
            }

            // double kappa = trans_tree->update_kappa(true_labels, class_count, true);
            double kappa = compute_kappa(predicted_labels, true_labels, class_count);
            // cout << "trans_tree kappa: " << trans_tree->kappa << endl;
            if (highest_kappa < trans_tree->kappa) {
                highest_kappa = trans_tree->kappa;
                matched_tree = trans_tree;
            }
        }
    }

    if (matched_tree == nullptr) {
        // cout << "match_concept: failed to match a tree" << endl;
        return nullptr;
    }

    // cout << "match_concept: matched a tree" << endl;
    return matched_tree;
}

double trans_pearl::compute_kappa(vector<int> predicted_labels, vector<int> actual_labels, int class_count) {
    // prepare confusion matrix
    vector<vector<int>> confusion_matrix(class_count, vector<int>(class_count, 0));
    int correct = 0;

    for (int i = 0; i < predicted_labels.size(); i++) {
        confusion_matrix[actual_labels[i]][predicted_labels[i]]++;
        if (actual_labels[i] == predicted_labels[i]) {
            correct++;
        }
    }

    double accuracy = (double) correct / predicted_labels.size();

    // computes the Cohen's kappa coefficient
    int sample_count = predicted_labels.size();
    double p0 = accuracy;
    double pc = 0.0;
    int row_count = class_count;
    int col_count = class_count;


    for (int i = 0; i < row_count; i++) {
        double row_sum = 0;
        for (int j = 0; j < col_count; j++) {
            row_sum += confusion_matrix[i][j];
        }

        double col_sum = 0;
        for (int j = 0; j < row_count; j++) {
            col_sum += confusion_matrix[j][i];
        }

        pc += (row_sum / sample_count) * (col_sum / sample_count);
    }

    if (pc == 1) {
        return 1;
    }

    return (p0 - pc) / (1.0 - pc);
}

int trans_pearl::evaluate_tree(shared_ptr<trans_pearl_tree> drifted_tree, vector<Instance*> &pseudo_instances) {
    // Evaluate drifted trees performances on pseudo instances generated by trees in other streams
    double correct_count = 0;
    for (auto instance : pseudo_instances) {
        if (drifted_tree->predict(*instance, false) != instance->getLabel()) continue;
        correct_count += 1;
    }

    return correct_count;
}

void trans_pearl::register_tree_pool(vector<shared_ptr<pearl_tree>>& _tree_pool) {
    this->registered_tree_pools.push_back(&_tree_pool);
}

vector<shared_ptr<pearl_tree>>& trans_pearl::get_concept_repo() {
    return this->tree_pool;
}

// ozaboost
// double trans_pearl::getEnsembleMemberWeight(int i) {
//     double em = this->swms[i] / (this->scms[i] + this->swms[i]);
//     if ((em == 0.0) || (em > 0.5)) {
//         return 0.0;
//     }
//     double Bm = em / (1.0 - em);
//     return log(1.0 / Bm);
// }

// int trans_pearl::predict() {
//     if (foreground_trees.empty()) {
//         init();
//     }
//
//     int num_classes = instance->getNumberClasses();
//     vector<int> votes(num_classes, 0);
//
//     for (int i = 0; i < num_trees; i++) {
//         int predicted_label = foreground_trees[i]->predict(*instance);
//         votes[predicted_label] += getEnsembleMemberWeight(i);
//     }
//
//     return vote(votes);
// }


// class trans_pearl_tree
trans_pearl_tree::trans_pearl_tree(int tree_pool_id,
                                   int kappa_window_size,
                                   double warning_delta,
                                   double drift_delta,
                                   std::mt19937 mrand,
                                   int instance_store_size)
        : pearl_tree(tree_pool_id,
                     kappa_window_size,
                     warning_delta,
                     drift_delta,
                     mrand),
         instance_store_size(instance_store_size) {}

trans_pearl_tree::trans_pearl_tree(trans_pearl_tree const &rhs)
        : pearl_tree(rhs.tree_pool_id,
                     rhs.kappa_window_size,
                     rhs.warning_delta,
                     rhs.drift_delta,
                     rhs.mrand),
          instance_store_size(instance_store_size) {}

void trans_pearl_tree::train(Instance& instance) {
    this->instance_store.push_back(&instance); // for generate_data
    if (this->bg_pearl_tree != nullptr) {
        shared_ptr<trans_pearl_tree> trans_bg_tree;
        trans_bg_tree = static_pointer_cast<trans_pearl_tree>(this->bg_pearl_tree);
        trans_bg_tree->instance_store.push_back(&instance);
        if (trans_bg_tree->instance_store.size() > this->instance_store_size) {
            trans_bg_tree->instance_store.erase(trans_bg_tree->instance_store.begin()); // TODO deque
        }
    }

    if (this->instance_store.size() > 500) {
        this->instance_store.erase(this->instance_store.begin()); // TODO deque
    }
    pearl_tree::train(instance);
}

vector<Instance*> trans_pearl_tree::generate_data(Instance* instance, int num_instances) {
    if (this->instance_store.size() < 10) {
        cout << "generate_data: not enough warning period data " << this->instance_store.size() << endl;
        return vector<Instance*>();
    }

    // cout << "generate_data..." << this->instance_store.size() << endl;
    vector<Instance*> pseudo_instances;

    for (int i = 0; i < num_instances; i++) {
        DenseInstance *pseudo_instance = this->tree->generate_data((DenseInstance *) instance);
        vector<DenseInstance*> close_instances = this->find_k_closest_instances(pseudo_instance,
                                                                          this->instance_store,
                                                                          1);

        // cout << "Before KNN-----------------------------" << endl;
        // for (double v : pseudo_instance->mInputData) {
        //     cout << v << " ";
        // }
        // for (double v : pseudo_instance->mOutputData) {
        //     cout << v << " ";
        // }
        // cout << endl;

        // Copy the rest of the attribute values to the pseudo instance
        for (Instance *close_instance : close_instances) {
            vector<int> modified_indices = pseudo_instance->modifiedAttIndices;
            for (int j = 0; j < close_instance->getNumberInputAttributes(); j++) {
                if (std::find(modified_indices.begin(), modified_indices.end(), j) != modified_indices.end()) {
                    continue;
                }
                pseudo_instance->setValue(j, close_instance->getInputAttributeValue(j));
            }
        }
        pseudo_instances.push_back(pseudo_instance);

        // cout << "After KNN-----------------------------" << endl;
        // for (double v : pseudo_instance->mInputData) {
        //     cout << v << " ";
        // }
        // for (double v : pseudo_instance->mOutputData) {
        //     cout << v << " ";
        // }
        // cout << endl;
    }

    return pseudo_instances;
}

vector<DenseInstance*> trans_pearl_tree::find_k_closest_instances(DenseInstance* target_instance,
                                                             vector<Instance*>& instance_store,
                                                             int k) {
    int num_row = target_instance->modifiedAttIndices.size() + 1;
    int num_col = instance_store.size();

    // Prepare data points
    vector<vector<double>> data(num_row, vector<double>());
    for (auto cur_instance : instance_store) {
        for (int i = 0; i < num_row - 1; i++) {
            int attIdx = target_instance->modifiedAttIndices[i];
            data[i].push_back(cur_instance->getInputAttributeValue(attIdx));
        }
        data[num_row - 1].push_back(cur_instance->getLabel());
    }

    Matrix dataPoints(num_row, num_col);
    for (int i = 0; i < num_row; i++) {
        dataPoints.row(i) = Eigen::VectorXd::Map(&data[i][0], data[i].size());
    }
    // cout << dataPoints << endl;

    knn::KDTreeMinkowski<double, knn::EuclideanDistance<double>> kdtree(dataPoints);
    kdtree.setBucketSize(16);
    kdtree.setCompact(false);
    kdtree.setBalanced(false);
    kdtree.setTakeRoot(true);
    kdtree.setMaxDistance(0);
    kdtree.setThreads(2);
    kdtree.build();

    vector<vector<double>> target_data(num_row, vector<double>());
    for (int i = 0; i < num_row - 1; i++) {
        int attIdx = target_instance->modifiedAttIndices[i];
        target_data[i].push_back(target_instance->getInputAttributeValue(attIdx));
    }
    target_data[num_row - 1].push_back(target_instance->getLabel());

    Matrix queryPoints(num_row, 1);
    for (int i = 0; i < num_row; i++) {
        queryPoints.row(i) = Eigen::VectorXd::Map(&target_data[i][0], target_data[i].size());
    }

    Matrixi indices;
    Matrix distances;
    kdtree.query(queryPoints, k, indices, distances);

    vector<DenseInstance*> close_instances;
    for (int i = 0; i < k; i++) {
        close_instances.push_back((DenseInstance*) instance_store[indices(i, 0)]);
    }

    return close_instances;
}


// class boosted_bg_tree_pool
trans_pearl::boosted_bg_tree_pool::boosted_bg_tree_pool(int pool_size,
                     int mini_batch_size,
                     shared_ptr<trans_pearl_tree> tree_template,
                     int lambda):
        mini_batch_size(mini_batch_size),
        pool_size(pool_size),
        tree_template(tree_template),
        lambda(lambda) {

    mrand = std::mt19937(42);
    oob_tree_correct_count.resize(mini_batch_size, 0);
    oob_tree_total_count.resize(mini_batch_size, 0);
}

void trans_pearl::boosted_bg_tree_pool::online_tradaboost(Instance *instance,
                                                          bool _is_same_distribution,
                                                          bool force_trigger) {
    if (mini_batch.size() == 0) {
        this->is_same_distribution = _is_same_distribution;
    } else if (this->is_same_distribution != _is_same_distribution) {
        cout << "online_tradaboost: inconsistent instance distributions" << endl;
        exit(1);
    }

    if (mini_batch.size() < mini_batch_size) {
        mini_batch.push_back(instance);
        if (!force_trigger) {
            return;
        }
    }

    update_bbt();
    boost();

    mini_batch.clear();
}

vector<shared_ptr<pearl_tree>> trans_pearl::boosted_bg_tree_pool::get_best_models() {
    vector<shared_ptr<pearl_tree>> best_models;
    for (auto tree : pool) {
        for (auto instance : warning_period_instances) {
            tree->predict(*instance, true);
        }

        // TODO return top k
        best_models.push_back(tree);
    }

    return best_models;
}

void trans_pearl::boosted_bg_tree_pool::update_bbt() {
    bbt_counter++;

    // create a new boosting tree for current mini-batch
    shared_ptr<trans_pearl_tree> new_tree = std::make_shared<trans_pearl_tree>(*tree_template);
    if (pool.size() < pool_size) {
        pool.push_back(new_tree);
    } else {
        pool[bbt_counter % pool_size] = new_tree;
    }
}

void trans_pearl::boosted_bg_tree_pool::boost() {
    for (Instance* instance : mini_batch) {
        double weight = 1.0;
        instance->setWeight(weight);
    }

    vector<int> lambdas(mini_batch_size, lambda);

    double z = 0;
    for (auto tree : pool) {
        for (int i = 0; i < mini_batch.size(); i++) {
            Instance* instance = mini_batch[i];

            // bagging
            std::poisson_distribution<int> poisson_distr(lambdas[i]);
            int k = poisson_distr(mrand);

            double weight = instance->getWeight();
            instance->setWeight(k * weight);
            tree->train(*instance);
            instance->setWeight(weight);

            // boosting based on out-of-bag errors
            if (k == 0) {
                oob_tree_total_count[i] += 1;
                if (tree->predict(*instance, false) == instance->getLabel()) {
                    oob_tree_correct_count[i] += 1;
                }

                double c = oob_tree_correct_count[i]
                                / oob_tree_total_count[i];
                if (is_same_distribution) {
                    lambdas[i] = 1 - c;
                } else {
                    lambdas[i] = c;
                }
            }
            z += lambdas[i];
        }

        for (int i = 0; i < mini_batch.size(); i++) {
            lambdas[i] = lambdas[i] / z;
        }
    }
}