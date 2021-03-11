#include "trans_tree.h"

trans_tree::trans_tree(
        int seed,
        int kappa_window_size,
        double warning_delta,
        double drift_delta,
        // transfer learning params
        int least_transfer_warning_period_instances_length,
        int instance_store_size,
        int num_diff_distr_instances,
        int bbt_pool_size,
        int eviction_interval,
        double transfer_kappa_threshold,
        string boost_mode_str) :
    kappa_window_size(kappa_window_size),
    warning_delta(warning_delta),
    drift_delta(drift_delta),
    least_transfer_warning_period_length(least_transfer_warning_period_instances_length),
    instance_store_size(instance_store_size),
    num_diff_distr_instances(num_diff_distr_instances),
    bbt_pool_size(bbt_pool_size),
    eviction_interval(eviction_interval),
    transfer_kappa_threshold(transfer_kappa_threshold) {

    mrand = std::mt19937(seed);

    if (boost_mode_map.find(boost_mode_str) == boost_mode_map.end() ) {
        cout << "Invalid boost mode" << endl;
        exit(1);
    }
    this->boost_mode = boost_mode_map[boost_mode_str];

}

void trans_tree::init() {
    foreground_tree = make_tree(0);
    tree_pool.push_back(foreground_tree);
}

shared_ptr<hoeffding_tree> trans_tree::make_tree(int tree_pool_id) {
    return make_shared<hoeffding_tree>(warning_delta, drift_delta);
}

void trans_tree::train() {
    if (foreground_tree == nullptr) {
        init();
    }

    int actual_label = instance->getLabel();
    if (actual_labels.size() >= kappa_window_size) {
        actual_labels.pop_front();
    }
    actual_labels.push_back(actual_label);

    // foreground_tree->store_instance(instance);
    // transfer(instance);

    foreground_tree->train(*instance);

    int predicted_label = foreground_tree->predict(*instance, true);
    int error_count = (int) (predicted_label != actual_label);

    bool warning_detected_only = false;
    bool drift_detected = false;

    // detect actual drift
    if (detect_change(error_count, foreground_tree->drift_detector)) {
        drift_detected = true;
        foreground_tree->warning_detector->resetChange();
        foreground_tree->drift_detector->resetChange();

        tree_pool.push_back(foreground_tree);
        foreground_tree = foreground_tree->bg_tree;

        if (bbt_pool != nullptr) {
            // TODO allow concept match after actual drift?
            if (bbt_pool->warning_period_instances.size() < least_transfer_warning_period_length) {
                // cout << "-------------------------------------warning_period_instances size is not enough: "
                //      << i << ":"
                //      << bbt_pools[i]->warning_period_instances.size() << endl;
                bbt_pool = nullptr;
            } else {
                shared_ptr<hoeffding_tree> matched_tree =
                        match_concept(bbt_pool->warning_period_instances);
                if (matched_tree == nullptr) {
                    bbt_pool = nullptr;
                } else {
                    bbt_pool->matched_tree = matched_tree;
                }
            }
        }
    }

    // detect warning
    if (detect_change(error_count, foreground_tree->warning_detector)) {
        foreground_tree->bg_tree = make_tree(-1);
        foreground_tree->warning_detector->resetChange();

        if (!drift_detected) {
            warning_detected_only = true;

            shared_ptr<hoeffding_tree> tree_template = foreground_tree->bg_tree;
            bbt_pool = make_unique<boosted_bg_tree_pool>(
                    boost_mode,
                    bbt_pool_size,
                    eviction_interval,
                    transfer_kappa_threshold,
                    tree_template,
                    1);
        }
    }
}

int trans_tree::predict() {
    if (foreground_tree == nullptr) {
        init();
    }

    return foreground_tree->predict(*instance);
}

bool trans_tree::detect_change(int error_count, unique_ptr<HT::ADWIN>& detector) {

    double old_error = detector->getEstimation();
    bool error_change = detector->setInput(error_count);

    if (!error_change) {
        return false;
    }

    if (old_error > detector->getEstimation()) {
        // error is decreasing
        return false;
    }

    return true;
}

bool trans_tree::init_data_source(const string& filename) {
    cout << "Initializing data source..." << endl;

    reader = make_unique<ArffReader>();
    if (!reader->setFile(filename)) {
        cout << "Failed to open file: " << filename << endl;
        exit(1);
    }

    return true;
}

bool trans_tree::get_next_instance() {
    if (!reader->hasNextInstance()) {
        return false;
    }

    instance = reader->nextInstance();
    return true;
}

int trans_tree::get_cur_instance_label() {
    return instance->getLabel();
}

void trans_tree::delete_cur_instance() {
    delete instance;
}


// class tree
hoeffding_tree::hoeffding_tree(double warning_delta, double drift_delta) :
        warning_delta(warning_delta),
        drift_delta(drift_delta) {

    tree = make_unique<HT::HoeffdingTree>();
    warning_detector = make_unique<HT::ADWIN>(warning_delta);
    drift_detector = make_unique<HT::ADWIN>(drift_delta);
    bg_tree = nullptr;
}

int hoeffding_tree::predict(Instance& instance) {
    double* classPredictions = tree->getPrediction(instance);
    int result = 0;
    double max_val = classPredictions[0];

    // Find class label with the highest probability
    for (int i = 1; i < instance.getNumberClasses(); i++) {
        if (max_val < classPredictions[i]) {
            max_val = classPredictions[i];
            result = i;
        }
    }

    return result;
}

void hoeffding_tree::train(Instance& instance) {
    tree->train(instance);

    if (bg_tree != nullptr) {
        bg_tree->train(instance);
    }
}
