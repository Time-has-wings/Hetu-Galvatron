import numpy as np
from typing import List

class DynamicProgramming:
    def __init__(
        self,
        layernum: int,
        strategy_num_list: List[int],
        layer_time_cost_list:List[List[float]],
        layer_memory_cost_list:List[List[int]],
        inter_layer_time_cost_list:List[List[List[float]]],
        extra_memory_cost:int,
        extra_time_cost:float,
        memory_constraint_in_MB:int,
        use_cpp_core:bool = True,
    ):
        # [Step 1] Check input validity
        assert layernum == len(strategy_num_list), f'layernum({layernum}) != strategy_num_list({len(strategy_num_list)})'
        assert layernum == len(layer_time_cost_list), f'layernum({layernum}) != layer_time_cost_list({len(layer_time_cost_list)})'
        assert layernum == len(layer_memory_cost_list), f'layernum({layernum}) != layer_memory_cost_list({len(layer_memory_cost_list)})'
        assert layernum == len(inter_layer_time_cost_list), f'layernum({layernum}) != inter_layer_time_cost_list({len(inter_layer_time_cost_list)})'

        for layer_idx in range(layernum):
            assert strategy_num_list[layer_idx] == len(layer_time_cost_list[layer_idx]), f'strategy_num_list[layer_idx]({strategy_num_list[layer_idx]}) != layer_time_cost_list[layer_idx]({len(layer_time_cost_list[layer_idx])})'
            assert strategy_num_list[layer_idx] == len(layer_memory_cost_list[layer_idx]), f'strategy_num_list[layer_idx]({strategy_num_list[layer_idx]}) != layer_memory_cost_list[layer_idx]({len(layer_memory_cost_list[layer_idx])})'

        for layer_idx in range(layernum):
            if layer_idx == 0:
                assert np.all(np.array(inter_layer_time_cost_list[layer_idx]) == 0), f"inter_layer_time_cost_list[{layer_idx}] should be all zeros, but got:\n{inter_layer_time_cost_list[layer_idx]}"
            else:
                switch_cost = inter_layer_time_cost_list[layer_idx]
                cur_layer_idx = layer_idx
                prev_layer_idx = layer_idx - 1
                assert len(switch_cost) == strategy_num_list[prev_layer_idx]
                assert len(switch_cost[0]) == strategy_num_list[cur_layer_idx]

        # [Step 2] Store parameters
        self.layernum = layernum
        self.strategy_num_list = strategy_num_list
        self.extra_memory_cost = int(extra_memory_cost)  # Convert to int for C++ compatibility
        self.extra_time_cost = extra_time_cost
        self.memory_constraint_in_MB = memory_constraint_in_MB + 1
        self.use_cpp_core = use_cpp_core

        # [Step 3] Convert layer_time_cost_list to NumPy array and pad to max_strategy_num. Note that the padding is done in the last dimension.
        # Calculate max_strategy_num for padding
        self.max_strategy_num = max(self.strategy_num_list)
        max_strategy_num = self.max_strategy_num
        
        # Convert layer_time_cost_list to NumPy array and pad to max_strategy_num
        # Shape: [layernum, max_strategy_num]
        self.layer_time_cost_list = np.full((layernum, max_strategy_num), np.inf, dtype=np.float64)
        for layer_idx in range(layernum):
            num_strategies = strategy_num_list[layer_idx]
            self.layer_time_cost_list[layer_idx, :num_strategies] = np.array(layer_time_cost_list[layer_idx], dtype=np.float64)
        
        # Convert layer_memory_cost_list to NumPy array and pad to max_strategy_num
        # Shape: [layernum, max_strategy_num]
        self.layer_memory_cost_list = np.full((layernum, max_strategy_num), 0, dtype=np.int32)
        for layer_idx in range(layernum):
            num_strategies = strategy_num_list[layer_idx]
            self.layer_memory_cost_list[layer_idx, :num_strategies] = np.array(layer_memory_cost_list[layer_idx], dtype=np.int32)
        
        # Convert inter_layer_time_cost_list to NumPy array and pad to max_strategy_num
        # Shape: [layernum, max_strategy_num, max_strategy_num]
        # Note: inter_layer_time_cost_list[layer][prev_strategy][cur_strategy]
        self.inter_layer_time_cost_list = np.zeros((layernum, max_strategy_num, max_strategy_num), dtype=np.float64)
        for layer_idx in range(layernum):
            if layer_idx == 0:
                # Layer 0 has no switching cost (already zeros)
                continue
            prev_strategies = strategy_num_list[layer_idx - 1]
            cur_strategies = strategy_num_list[layer_idx]
            # Copy the switching costs from the input list
            for prev_strategy_idx in range(prev_strategies):
                for cur_strategy_idx in range(cur_strategies):
                    self.inter_layer_time_cost_list[layer_idx, prev_strategy_idx, cur_strategy_idx] = \
                        inter_layer_time_cost_list[layer_idx][prev_strategy_idx][cur_strategy_idx]

        # [Step 4] Initialize DP arrays
        # Initialize 3D arrays: [layer][memory][strategy]
        self.f = np.full((self.layernum, self.memory_constraint_in_MB, max_strategy_num), np.inf, dtype=np.float64)
        # Use two separate arrays for path to store (prev_memory, prev_strategy)
        self.path_memory = np.full((self.layernum, self.memory_constraint_in_MB, max_strategy_num), -1, dtype=np.int32)
        self.path_strategy = np.full((self.layernum, self.memory_constraint_in_MB, max_strategy_num), -1, dtype=np.int32)
        
        # Store the best result
        self.best_time = np.inf
        self.best_memory = -1
        self.best_strategy_of_last_layer = -1
        self.best_strategy_list = None

    def fit(self):
        """
        Solve the dynamic programming problem to find the minimum time cost
        under memory constraints.
        """

        if self.use_cpp_core:
            import dynamic_program_cpp
            # Pass NumPy arrays directly to C++ backend (pybind11 will handle the conversion efficiently)
            # This avoids data copying and improves performance
            # Also pass f, path_memory, path_strategy arrays to avoid reallocation in C++
            best_time, strategy_list, best_memory = dynamic_program_cpp.dynamic_programming_fit(
                layernum=self.layernum,
                max_strategy_num=self.max_strategy_num,
                extra_memory_cost=self.extra_memory_cost,
                extra_time_cost=self.extra_time_cost,
                memory_constraint_in_MB=self.memory_constraint_in_MB,
                strategy_num_list=self.strategy_num_list,
                layer_time_cost_list=self.layer_time_cost_list,
                layer_memory_cost_list=self.layer_memory_cost_list,
                inter_layer_time_cost_list=self.inter_layer_time_cost_list,
                f=self.f,
                path_memory=self.path_memory,
                path_strategy=self.path_strategy,
            )
            # Convert empty list to None for consistency with Python backend
            if strategy_list == []:
                strategy_list = None
            return best_time, strategy_list, best_memory
        else:
            return self._fit_python_core()

    def _fit_python_core(self):
        """
            f[i][m][s] = minimum time cost after processing first i layers using memory m, 
                         where the i-th layer uses strategy s
            path[i][m][s] = (prev_memory, prev_strategy) for backtracking
        """
        # Initialize layer 0 (no switching overhead)
        for strategy_idx in range(self.strategy_num_list[0]):
            memory_cost = int(self.layer_memory_cost_list[0][strategy_idx] + self.extra_memory_cost)
            time_cost = self.layer_time_cost_list[0][strategy_idx] + self.extra_time_cost
            
            if memory_cost < self.memory_constraint_in_MB:
                self.f[0][memory_cost][strategy_idx] = time_cost
                self.path_memory[0][memory_cost][strategy_idx] = -1
                self.path_strategy[0][memory_cost][strategy_idx] = -1
        
        # Process remaining layers
        for layer_idx in range(1, self.layernum):
            cur_layer_idx = layer_idx
            prev_layer_idx = layer_idx - 1
            
            # try all possible strategies for the current layer
            for cur_strategy_idx in range(self.strategy_num_list[cur_layer_idx]):
                cur_memory_cost = self.layer_memory_cost_list[cur_layer_idx][cur_strategy_idx]
                cur_time_cost = self.layer_time_cost_list[cur_layer_idx][cur_strategy_idx]
                
                # try all possible strategies for the previous layer
                for prev_strategy_idx in range(self.strategy_num_list[prev_layer_idx]):
                    switch_cost = self.inter_layer_time_cost_list[cur_layer_idx][prev_strategy_idx][cur_strategy_idx]
                    
                    # Try all possible memory states from previous layer
                    for prev_memory in range(self.memory_constraint_in_MB):
                        if self.f[prev_layer_idx][prev_memory][prev_strategy_idx] == np.inf:
                            continue
                        
                        # Calculate new memory usage
                        new_memory = int(prev_memory + cur_memory_cost)
                        if new_memory >= self.memory_constraint_in_MB:
                            continue
                        
                        # Calculate new time cost
                        new_time = self.f[prev_layer_idx][prev_memory][prev_strategy_idx] + cur_time_cost + switch_cost
                        
                        # Update if better
                        if new_time < self.f[cur_layer_idx][new_memory][cur_strategy_idx]:
                            self.f[cur_layer_idx][new_memory][cur_strategy_idx] = new_time
                            self.path_memory[cur_layer_idx][new_memory][cur_strategy_idx] = prev_memory
                            self.path_strategy[cur_layer_idx][new_memory][cur_strategy_idx] = prev_strategy_idx
        
        # Find the best solution in the last layer
        for memory in range(self.memory_constraint_in_MB):
            for strategy_idx in range(self.strategy_num_list[self.layernum - 1]):
                if self.f[self.layernum - 1][memory][strategy_idx] < self.best_time:
                    self.best_time = self.f[self.layernum - 1][memory][strategy_idx]
                    self.best_memory = memory
                    self.best_strategy_of_last_layer = strategy_idx
        
        # Backtrack to find the strategy sequence
        if self.best_time != np.inf:
            self.best_strategy_list = self._backtrack()
    
        return self.best_time, self.best_strategy_list, self.best_memory

    def _backtrack(self):
        """
        Backtrack to find the optimal strategy sequence.
        Returns a list where result[i] is the strategy index chosen at layer i.
        """
        if self.best_time == np.inf:
            return None
        
        strategy_list = [0] * self.layernum
        current_memory = self.best_memory
        current_strategy = self.best_strategy_of_last_layer
        
        # Backtrack from the last layer
        for layer_idx in range(self.layernum - 1, -1, -1):
            prev_memory = self.path_memory[layer_idx][current_memory][current_strategy]
            prev_strategy = self.path_strategy[layer_idx][current_memory][current_strategy]
            
            strategy_list[layer_idx] = current_strategy
            
            # Stop backtracking at layer 0
            if layer_idx == 0:
                break
            
            current_memory = prev_memory
            current_strategy = prev_strategy
        
        return strategy_list