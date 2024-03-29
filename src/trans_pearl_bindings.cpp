#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <PEARL/src/cpp/pearl.h>
#include "trans_pearl.h"
#include "trans_pearl_wrapper.h"
#include "trans_tree_wrapper.h"

// PYBIND11_MAKE_OPAQUE(vector<Instance*>);
// PYBIND11_MAKE_OPAQUE(vector<shared_ptr<pearl_tree>>);

namespace py = pybind11;

PYBIND11_MODULE(trans_pearl_wrapper, m) {
    m.doc() = "trans_pearl's implementation in C++";

    // py::class_<std::vector<Instance*>>(m, "IntInstance")
    //         .def(py::init<>())
    //         .def("clear", &std::vector<Instance*>::clear)
    //         .def("pop_back", &std::vector<Instance*>::pop_back)
    //         .def("__len__", [](const std::vector<Instance*> &v) { return v.size(); });

    // py::class_<vector<shared_ptr<pearl_tree>>>(m, "pearl_tree_pool")
    //         .def(py::init<>())
    //         .def("clear", &std::vector<shared_ptr<pearl_tree>>::clear)
    //         .def("pop_back", &std::vector<shared_ptr<pearl_tree>>::pop_back)
    //         .def("__len__", [](const std::vector<shared_ptr<pearl_tree>> &v) { return v.size(); });

    py::class_<adaptive_random_forest>(m, "adaptive_random_forest")
        .def(py::init<int,
                      int,
                      int,
                      int,
                      double,
                      double>())
        .def("init_data_source", &adaptive_random_forest::init_data_source)
        .def("get_next_instance", &adaptive_random_forest::get_next_instance)
        .def("get_cur_instance_label", &adaptive_random_forest::get_cur_instance_label)
        .def("delete_cur_instance", &adaptive_random_forest::delete_cur_instance)
        .def("predict", &adaptive_random_forest::predict)
        .def("train", &adaptive_random_forest::train);

    py::class_<pearl, adaptive_random_forest>(m, "pearl")
        .def(py::init<int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      double,
                      double,
                      double,
                      double,
                      double,
                      bool,
                      bool>())
        .def("get_candidate_tree_group_size", &pearl::get_candidate_tree_group_size)
        .def("get_tree_pool_size", &pearl::get_tree_pool_size)
        .def("is_state_graph_stable", &pearl::is_state_graph_stable)
        .def("__repr__",
            [](const pearl &p) {
                return "<pearl.pearl has "
                    + std::to_string(p.get_tree_pool_size()) + " trees>";
            }
         );

    // py::class_<trans_pearl, pearl>(m, "trans_pearl")
    //     .def(py::init<int,
    //                   int,
    //                   int,
    //                   int,
    //                   int,
    //                   int,
    //                   int,
    //                   int,
    //                   int,
    //                   int,
    //                   double,
    //                   double,
    //                   double,
    //                   double,
    //                   double,
    //                   int,
    //                   double,
    //                   int,
    //                   double>())
    //     .def("select_candidate_trees", &pearl::select_candidate_trees)
    //     .def("select_predicted_trees", &trans_pearl::select_predicted_trees)
    //     .def("has_actual_drift", &trans_pearl::has_actual_drift)
    //     .def("find_last_actual_drift_point", &trans_pearl::find_last_actual_drift_point)
    //     .def("train", &trans_pearl::train)
    //     .def("adapt_state", &trans_pearl::adapt_state)
    //     .def("adapt_state_with_proactivity", &trans_pearl::adapt_state_with_proactivity)
    //     .def("update_drifted_tree_indices", &trans_pearl::update_drifted_tree_indices)
    //     .def("set_expected_drift_prob", &trans_pearl::set_expected_drift_prob)
    //     .def("get_stable_tree_indices", &trans_pearl::get_stable_tree_indices)
    //     .def("has_actual_drifted_trees", &trans_pearl::has_actual_drifted_trees)
    //     .def("get_concept_repo", &trans_pearl::get_concept_repo, pybind11::return_value_policy::reference)
    //     .def("get_concept_repo", &trans_pearl::get_concept_repo)
    //     .def("match_concept", &trans_pearl::match_concept)
    //     .def("transfer", &trans_pearl::transfer)
    //     .def("register_tree_pool", &trans_pearl::register_tree_pool);

        py::class_<trans_pearl_wrapper>(m, "trans_pearl_wrapper")
                .def(py::init<int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        double,
                        double,
                        double,
                        double,
                        double,
                        int,
                        int,
                        int,
                        int,
                        int,
                        double,
                        string>())
                .def(py::init<int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        double,
                        double,
                        double,
                        double,
                        double>())
                .def("switch_classifier", &trans_pearl_wrapper::switch_classifier)
                .def("train", &trans_pearl_wrapper::train)
                .def("predict", &trans_pearl_wrapper::predict)
                .def("get_cur_instance_label", &trans_pearl_wrapper::get_cur_instance_label)
                .def("init_data_source", &trans_pearl_wrapper::init_data_source)
                .def("get_next_instance", &trans_pearl_wrapper::get_next_instance)
                .def("get_candidate_tree_group_size", &trans_pearl_wrapper::get_candidate_tree_group_size)
                .def("get_transferred_tree_group_size", &trans_pearl_wrapper::get_transferred_tree_group_size)
                .def("get_tree_pool_size", &trans_pearl_wrapper::get_tree_pool_size);



    py::class_<trans_tree_wrapper>(m, "trans_tree_wrapper")
            .def(py::init<
                    int,
                    int,
                    int,
                    double,
                    double,
                    // transfer learning params
                    int,
                    int,
                    int,
                    int,
                    int,
                    double,
                    double,
                    double,
                    string>())
            .def("init_data_source", &trans_tree_wrapper::init_data_source)
            .def("get_next_instance", &trans_tree_wrapper::get_next_instance)
            .def("get_cur_instance_label", &trans_tree_wrapper::get_cur_instance_label)
            .def("predict", &trans_tree_wrapper::predict)
            .def("train", &trans_tree_wrapper::train)
            .def("switch_classifier", &trans_tree_wrapper::switch_classifier)
            .def("train", &trans_tree_wrapper::train)
            .def("predict", &trans_tree_wrapper::predict)
            .def("get_cur_instance_label", &trans_tree_wrapper::get_cur_instance_label)
            .def("init_data_source", &trans_tree_wrapper::init_data_source)
            .def("get_next_instance", &trans_tree_wrapper::get_next_instance)
            .def("get_transferred_tree_group_size", &trans_tree_wrapper::get_transferred_tree_group_size)
            .def("get_tree_pool_size", &trans_tree_wrapper::get_tree_pool_size);

}
