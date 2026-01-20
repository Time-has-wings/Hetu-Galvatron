#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <limits>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace py = pybind11;

// Helper functions for array indexing
inline int get_f_index(int layer_idx, int memory, int strategy_idx, int memory_constraint_in_MB, int max_strategy_num) {
    return layer_idx * memory_constraint_in_MB * max_strategy_num + memory * max_strategy_num + strategy_idx;
}

inline int get_intra_cost_index(int layer_idx, int strategy_idx, int max_strategy_num) {
    return layer_idx * max_strategy_num + strategy_idx;
}

inline int get_inter_cost_index(int layer_idx, int prev_strategy_idx, int cur_strategy_idx, int max_strategy_num) {
    return layer_idx * max_strategy_num * max_strategy_num + prev_strategy_idx * max_strategy_num + cur_strategy_idx;
}

// Main dynamic programming function
std::tuple<double, std::vector<int>, int> dynamic_programming_fit(
    int layernum,
    int max_strategy_num,
    int extra_memory_cost,
    double extra_time_cost,
    int memory_constraint_in_MB,
    const std::vector<int>& strategy_num_list,
    py::array_t<double> layer_time_cost_list,
    py::array_t<int> layer_memory_cost_list,
    py::array_t<double> inter_layer_time_cost_list,
    py::array_t<double> f,
    py::array_t<int> path_memory,
    py::array_t<int> path_strategy
) {
    // Get buffer info and pointers
    py::buffer_info layer_time_cost_info = layer_time_cost_list.request();
    double* layer_time_cost_ptr = static_cast<double*>(layer_time_cost_info.ptr);

    py::buffer_info layer_memory_cost_info = layer_memory_cost_list.request();
    int* layer_memory_cost_ptr = static_cast<int*>(layer_memory_cost_info.ptr);

    py::buffer_info inter_layer_time_cost_info = inter_layer_time_cost_list.request();
    double* inter_layer_time_cost_ptr = static_cast<double*>(inter_layer_time_cost_info.ptr);

    py::buffer_info f_info = f.request();
    double* f_ptr = static_cast<double*>(f_info.ptr);

    py::buffer_info path_memory_info = path_memory.request();
    int* path_memory_ptr = static_cast<int*>(path_memory_info.ptr);

    py::buffer_info path_strategy_info = path_strategy.request();
    int* path_strategy_ptr = static_cast<int*>(path_strategy_info.ptr);
    
    // Initialize best solution tracking
    double best_time = std::numeric_limits<double>::infinity();
    int best_memory = -1;
    int best_strategy_of_last_layer = -1;

    // Initialize layer 0 (no switching overhead)
    for (int strategy_idx = 0; strategy_idx < strategy_num_list[0]; ++strategy_idx) {
        int intra_idx = get_intra_cost_index(0, strategy_idx, max_strategy_num);
        int memory_cost = layer_memory_cost_ptr[intra_idx] + extra_memory_cost;
        double time_cost = layer_time_cost_ptr[intra_idx] + extra_time_cost;
        
        if (memory_cost < memory_constraint_in_MB) {
            int f_idx = get_f_index(0, memory_cost, strategy_idx, memory_constraint_in_MB, max_strategy_num);
            f_ptr[f_idx] = time_cost;
            path_memory_ptr[f_idx] = -1;
            path_strategy_ptr[f_idx] = -1;
        }
    }

    // Process remaining layers
    for (int layer_idx = 1; layer_idx < layernum; ++layer_idx) {
        int prev_layer_idx = layer_idx - 1;
        int prev_strategies = strategy_num_list[prev_layer_idx];
        int cur_strategies = strategy_num_list[layer_idx];
        
        // try all possible strategies for the current layer
        for (int cur_strategy_idx = 0; cur_strategy_idx < cur_strategies; ++cur_strategy_idx) {
            int intra_idx = get_intra_cost_index(layer_idx, cur_strategy_idx, max_strategy_num);
            int cur_memory_cost = layer_memory_cost_ptr[intra_idx];
            double cur_time_cost = layer_time_cost_ptr[intra_idx];
            
            for (int prev_strategy_idx = 0; prev_strategy_idx < prev_strategies; ++prev_strategy_idx) {
                int inter_idx = get_inter_cost_index(layer_idx, prev_strategy_idx, cur_strategy_idx, max_strategy_num);
                double switch_cost = inter_layer_time_cost_ptr[inter_idx];
                
                for (int prev_memory = 0; prev_memory < memory_constraint_in_MB; ++prev_memory) {
                    int prev_f_idx = get_f_index(prev_layer_idx, prev_memory, prev_strategy_idx, memory_constraint_in_MB, max_strategy_num);
                    if (std::isinf(f_ptr[prev_f_idx])) {
                        continue;
                    }
                    
                    // Calculate new memory usage
                    int new_memory = prev_memory + cur_memory_cost;
                    if (new_memory >= memory_constraint_in_MB) {
                        continue;
                    }
                    
                    // Calculate new time cost
                    double new_time = f_ptr[prev_f_idx] + cur_time_cost + switch_cost;
                    
                    // Update if better
                    int cur_f_idx = get_f_index(layer_idx, new_memory, cur_strategy_idx, memory_constraint_in_MB, max_strategy_num);
                    if (new_time < f_ptr[cur_f_idx]) {
                        f_ptr[cur_f_idx] = new_time;
                        path_memory_ptr[cur_f_idx] = prev_memory;
                        path_strategy_ptr[cur_f_idx] = prev_strategy_idx;
                    }
                }
            }
        }
    }

    // Find the best solution in the last layer
    for (int memory = 0; memory < memory_constraint_in_MB; ++memory) {
        for (int strategy_idx = 0; strategy_idx < strategy_num_list[layernum - 1]; ++strategy_idx) {
            int f_idx = get_f_index(layernum - 1, memory, strategy_idx, memory_constraint_in_MB, max_strategy_num);
            double time_val = f_ptr[f_idx];
            if (time_val < best_time) {
                best_time = time_val;
                best_memory = memory;
                best_strategy_of_last_layer = strategy_idx;
            }
        }
    }
    
    // Backtrack to find the strategy sequence
    if (best_time != std::numeric_limits<double>::infinity()) {
        std::vector<int> best_strategy_list(layernum, 0);
        int current_memory = best_memory;
        int current_strategy = best_strategy_of_last_layer;
        
        // Backtrack from the last layer
        for (int layer_idx = layernum - 1; layer_idx >= 0; --layer_idx) {
            int idx = get_f_index(layer_idx, current_memory, current_strategy, memory_constraint_in_MB, max_strategy_num);
            int prev_memory = path_memory_ptr[idx];
            int prev_strategy = path_strategy_ptr[idx];
            
            best_strategy_list[layer_idx] = current_strategy;
            
            // Stop backtracking at layer 0
            if (layer_idx == 0) {
                break;
            }
        
            current_memory = prev_memory;
            current_strategy = prev_strategy;
        }
        return std::make_tuple(best_time, best_strategy_list, best_memory);
    }
    else {
        return std::make_tuple(best_time, std::vector<int>(), best_memory);
    }
}

PYBIND11_MODULE(dynamic_program_cpp, m) {
    m.def("dynamic_programming_fit", &dynamic_programming_fit,
          "Solve the dynamic programming problem",
          py::arg("layernum"),
          py::arg("max_strategy_num"),
          py::arg("extra_memory_cost"),
          py::arg("extra_time_cost"),
          py::arg("memory_constraint_in_MB"),
          py::arg("strategy_num_list"),
          py::arg("layer_time_cost_list"),
          py::arg("layer_memory_cost_list"),
          py::arg("inter_layer_time_cost_list"),
          py::arg("f"),
          py::arg("path_memory"),
          py::arg("path_strategy"));
}
