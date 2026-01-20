from ..cost_model_args import TrainArgs, ProfileHardwareArgs
from logging import Logger
from types import SimpleNamespace
from galvatron.utils.strategy_utils import byte_to_MB

class PipelineTimeCostModel:
    pipeline_time_args_list = {
        'TrainArgs': ['seq_length', 'hidden_size', 'mixed_precision'],
        'ProfileHardwareArgs': ['p2p_comm_coe_dict'],
    }

    def __init__(
        self,
        global_batch_size:int = 8,
        chunks: int = 1,
        pp_size: int = 1,
        world_size:int = 8,
        logger: Logger = None,
        train_args: TrainArgs = None,
        profile_hardware_args: ProfileHardwareArgs = None,
    ):
        self.global_batch_size = global_batch_size
        self.chunks = chunks
        self.pp_size = pp_size
        self.world_size = world_size
        self.logger = logger
        
        self.args = SimpleNamespace()
        components = {'TrainArgs': train_args, 'ProfileHardwareArgs': profile_hardware_args}
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in self.pipeline_time_args_list[class_name]:
                    setattr(self.args, key, value)
        
        self.initialize()
        
    def initialize(self):
        args = self.args

        # [Step 1] copy some attributes for easy access
        self.seq_length = args.seq_length
        self.hidden_size = args.hidden_size

        # [Step 2] calculate 
        total_token_per_chunk = self.global_batch_size // self.chunks * self.seq_length
        dp_tp_sp_cp_size = self.world_size // self.pp_size
        self.token_per_chunk_per_rank = total_token_per_chunk // dp_tp_sp_cp_size
        
        # [Step 3] calculate pipeline time
        if self.pp_size == 1:
            self.p2p_communication_time = 0 
        else:
            self.p2p_comm_coe = args.p2p_comm_coe_dict[self.pp_size]

            dtype_size = 2 if args.mixed_precision else 4
            message_size_in_MB = self.token_per_chunk_per_rank * self.hidden_size * dtype_size / byte_to_MB

            # forward: send hidden_states. backward: send gradient.
            self.p2p_message_size_in_MB = message_size_in_MB * 2
            self.p2p_communication_time = self.p2p_message_size_in_MB * self.p2p_comm_coe
        
    def gen_result(self):
        return self.p2p_communication_time