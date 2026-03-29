import os


class BaseProfiler():
    def __init__(self):
        self.work_dir = None
        self.model_name = None
        self.profile_unit = None
        self.mixed_precision = None
        self.specific_time_path = None
        self.specific_memory_path = None

    def set_work_dir(self, work_dir):
        self.work_dir = work_dir

    def set_model_name(self, model_name):
        self.model_name = model_name

    def set_profile_unit(self, profile_unit):
        self.profile_unit = profile_unit

    def set_mixed_precision(self, mixed_precision):
        self.mixed_precision = mixed_precision

    def set_specific_time_path(self, specific_time_path):
        self.specific_time_path = specific_time_path

    def set_specific_memory_path(self, specific_memory_path):
        self.specific_memory_path = specific_memory_path

    def memory_profiling_path(self):
        """Get memory profiling path

        Returns:
            str: Path to memory profiling config file
        """
        if self.specific_memory_path is not None:
            return self.specific_memory_path
        
        assert self.work_dir is not None, "Should specify the work directory!"
        assert self.model_name is not None, "Should specify the model name!"
        assert self.profile_unit is not None, "Should specify the profile unit!"
        assert self.mixed_precision is not None, "Should specify the mixed precision!"

        memory_config_path = f'configs/memory_profiling_{self.mixed_precision}_{self.model_name}_{self.profile_unit}.json'
        return os.path.join(self.work_dir, memory_config_path)

    def time_profiling_path(self):
        """Get time profiling path

        Returns:
            str: Path to time profiling config file
        """
        if self.specific_time_path is not None:
            return self.specific_time_path
        
        assert self.work_dir is not None, "Should specify the work directory!"
        assert self.model_name is not None, "Should specify the model name!"
        assert self.profile_unit is not None, "Should specify the profile unit!"
        assert self.mixed_precision is not None, "Should specify the mixed precision!"
        
        time_config_path = f'configs/computation_profiling_{self.mixed_precision}_{self.model_name}_{self.profile_unit}.json'
        return os.path.join(self.work_dir, time_config_path)
