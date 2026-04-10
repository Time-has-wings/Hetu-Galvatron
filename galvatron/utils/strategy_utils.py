from dataclasses import dataclass
from enum import Enum
from typing import List, Union

from .print_utils import ColorSet


byte_to_MB = 1024 * 1024
model_states_to_param_size_ratio = 4

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

class DPType(Enum):
    DDP = 'ddp'
    ZERO2 = 'zero2'
    ZERO3 = 'zero3'
    
    @classmethod
    def values(cls):
        return [item for item in cls]
    
    @classmethod
    def contains(cls, value) -> bool:
        return value in cls.values()

    def __lt__(self, other):
        if not isinstance(other, DPType):
            raise TypeError(f"Cannot compare '{type(self)}' and '{type(other)}' types")
        return self.value < other.value

@dataclass
class StrategyBase:
    pass

@dataclass
class EmbeddingLMHeadStrategy(StrategyBase):
    pp_size: int = 1
    tp_size: int = 1
    sp_size: int = 1
    cp_size: int = 1
    dp_size: int = 1
    dp_type: DPType = DPType.ZERO2

    def __post_init__(self):
        self._check_and_fix_sdp()
        self._check_tp_sp()
    
    def _check_and_fix_sdp(self):
        if self.sdp_size == 1 and self.dp_type != DPType.DDP:
            print(f"{ColorSet.YELLOW}[WARNING] [{self.__class__.__name__}] When sdp_size is 1, dp_type should be 'DPType.DDP'. Got '{self.dp_type}' instead. Automatically resetting to 'DPType.DDP'.{ColorSet.RESET}")
            self.dp_type = DPType.DDP

    def _check_tp_sp(self):
        assert not (self.tp_size > 1 and self.sp_size > 1), f"{ColorSet.RED}[ERROR] [{self.__class__.__name__}] TP and SP cannot be used together. Got tp_size={self.tp_size} and sp_size={self.sp_size}.{ColorSet.RESET}"

    @property
    def world_size(self):
        return self.pp_size * self.tp_size * self.sp_size * self.cp_size * self.dp_size

    @property
    def sdp_size(self):
        return self.dp_size * self.sp_size * self.cp_size
    
    @property
    def tp_sp_size(self):
        return max(self.tp_size, self.sp_size)

    def to_string(self):
        return f"[{self.__class__.__name__}]({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"

    def to_simple_string(self):
        string = f'{self.pp_size}-'

        if self.tp_sp_size != 1:
            string += f'{self.tp_sp_size}*-'
        else:
            string += f'{self.tp_sp_size}-'

        if self.dp_type == DPType.ZERO3:
            string += f'{self.dp_size}f'
        else:
            string += f'{self.dp_size}'

        if hasattr(self, 'checkpoint') and self.checkpoint:
            string += '-c'
        
        if self.sp_size > 1:
            string += '-sp'
        
        return string
    
    def __eq__(self, other):
        if type(other) != type(self):
            return False
        for field in self.__dataclass_fields__:
            if getattr(self, field) != getattr(other, field):
                return False
        return True
    
    def __lt__(self, other):
        if type(other) != type(self):
            return NotImplemented
        for field in self.__dataclass_fields__:
            if getattr(self, field) < getattr(other, field):
                return True
            elif getattr(self, field) > getattr(other, field):
                return False
        return False

    def __hash__(self):
        attrs = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        return hash(attrs)
    
    def __str__(self):
        return f"[{self.__class__.__name__}]({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"

@dataclass
class AttentionStrategy(EmbeddingLMHeadStrategy):
    checkpoint: bool = False

    def __hash__(self):
        attrs = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        return hash(attrs)
    
    def to_embedding_lmhead_strategy(self):
        return EmbeddingLMHeadStrategy(
            pp_size=self.pp_size,
            tp_size=self.tp_size,
            sp_size=self.sp_size,
            cp_size=self.cp_size,
            dp_size=self.dp_size,
            dp_type=self.dp_type
        )

    def to_ffn_strategy(self):
        return FFNStrategy(
            pp_size=self.pp_size,
            tp_size=self.tp_size,
            sp_size=self.sp_size,
            cp_size=self.cp_size,
            dp_size=self.dp_size,
            dp_type=self.dp_type,
            checkpoint=self.checkpoint
        )

    def to_layer_strategy(self):
        return LayerStrategy(
            pp_size=self.pp_size,
            tp_size=self.tp_size,
            sp_size=self.sp_size,
            cp_size=self.cp_size,
            dp_size=self.dp_size,
            dp_type=self.dp_type,
            checkpoint=self.checkpoint
        )


@dataclass
class FFNStrategy(EmbeddingLMHeadStrategy):
    checkpoint: bool = False

    def __hash__(self):
        attrs = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        return hash(attrs)
    
    def to_embedding_lmhead_strategy(self):
        return EmbeddingLMHeadStrategy(
            pp_size=self.pp_size,
            tp_size=self.tp_size,
            sp_size=self.sp_size,
            cp_size=self.cp_size,
            dp_size=self.dp_size,
            dp_type=self.dp_type
        )

@dataclass
class LayerStrategy(EmbeddingLMHeadStrategy):
    checkpoint: bool = False

    def __hash__(self):
        attrs = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        return hash(attrs)

    def to_embedding_lmhead_strategy(self):
        return EmbeddingLMHeadStrategy(
            pp_size=self.pp_size,
            tp_size=self.tp_size,
            sp_size=self.sp_size,
            cp_size=self.cp_size,
            dp_size=self.dp_size,
            dp_type=self.dp_type
        )

@dataclass
class MoEFFNStrategy(StrategyBase):
    pp_size: int = 1
    ep_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    dp_type: DPType = DPType.ZERO2
    checkpoint: bool = False

    def __post_init__(self):
        self._check_and_fix_dp()

    def _check_and_fix_dp(self):
        if self.dp_size > 1:
            assert DPType.contains(self.dp_type), f"{ColorSet.RED}[ERROR] [{self.__class__.__name__}] When dp_size > 1, strategy.dp_type must be in {DPType.values()}, but got '{self.dp_type}'.{ColorSet.RESET}"
        elif self.dp_size == 1 and self.dp_type != DPType.DDP:
            print(f"{ColorSet.YELLOW}[WARNING] [{self.__class__.__name__}] When dp_size is 1, dp_type should be 'DPType.DDP'. Got '{self.dp_type}' instead. Automatically resetting to 'DPType.DDP'.{ColorSet.RESET}")
            self.dp_type = DPType.DDP
    
    @property
    def world_size(self):
        return self.pp_size * self.tp_size * self.dp_size * self.ep_size

    @property
    def sdp_size(self):
        return self.dp_size

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        for field in self.__dataclass_fields__:
            if getattr(self, field) != getattr(other, field):
                return False
        return True
    
    def __lt__(self, other):
        if type(other) != type(self):
            return NotImplemented
        for field in self.__dataclass_fields__:
            if getattr(self, field) < getattr(other, field):
                return True
            elif getattr(self, field) > getattr(other, field):
                return False
        return False

    def __hash__(self):
        attrs = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        return hash(attrs)
    
    def __str__(self):
        return f"[{self.__class__.__name__}]({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"


def old_version_strategy_to_new_version_strategy(strategy:list, default_dp_type:str):
    pp_size = strategy[0]
    tp_size = strategy[1]
    dp_size = strategy[2]
    fix_cp_size = 1 # cp size fix to 1

    info = strategy[-1]
    use_ulysses = True if 'sp' in info.keys() and info['sp'] == 1 else False
    if use_ulysses:
        tp_size, sp_size = 1, tp_size
    else:
        tp_size, sp_size = tp_size, 1
    checkpoint = True if 'cpt' in info.keys() and info['cpt'] == 1 else False
    use_fsdp = True if 'fsdp' in info.keys() and info['fsdp'] == 1 else False
    dp_type = DPType.ZERO3 if use_fsdp else DPType.DDP if default_dp_type == 'ddp' else DPType.ZERO2
    if dp_size == 1:
        dp_type = DPType.DDP

    strategy:LayerStrategy = LayerStrategy(
        pp_size=pp_size,
        tp_size=tp_size,
        sp_size=sp_size,
        cp_size=fix_cp_size,
        dp_size=dp_size,
        dp_type=dp_type,
        checkpoint=checkpoint
    )
    return strategy

def new_version_strategy_to_old_version_strategy(strategy:StrategyBase):
    info = {}
    if strategy.dp_size > 1:
        if strategy.dp_type == DPType.ZERO3:
            info['fsdp'] = 1
        else:
            info['fsdp'] = 0
    
    if max(strategy.tp_size, strategy.sp_size) > 1:
        info['tp'] = 1
        if strategy.sp_size > 1:
            info['sp'] = 1
        else:
            info['sp'] = 0

    if strategy.checkpoint:
        info['cpt'] = 1

    pp_size = strategy.pp_size
    tp_size = max(strategy.tp_size, strategy.sp_size)
    dp_size = strategy.dp_size
    return [pp_size, tp_size, dp_size, info]

def print_strategy_list(strategy_list:Union[List[LayerStrategy], List[EmbeddingLMHeadStrategy], None], logger=None):
    if strategy_list is not None:
        string_list = [strategy.to_simple_string() for strategy in strategy_list]
        if logger is None:
            print(', '.join(string_list))
        else:
            logger.info(', '.join(string_list))

def strategy_list2config(strategy_list:List[LayerStrategy]):
    layer_num = len(strategy_list)
    if layer_num == 0:
        return {}

    pp_size = strategy_list[0].pp_size
    tp_sizes_enc = ','.join([str(strategy.tp_sp_size) for strategy in strategy_list])
    tp_consecutive_flags = ','.join(['1' for _ in range(layer_num)])
    dp_types_enc = ','.join(['1' if strategy.dp_type == DPType.ZERO3 else '0' for strategy in strategy_list])
    sp = ','.join(['1' if strategy.sp_size > 1 else '0' for strategy in strategy_list])
    checkpoint = ','.join(['1' if strategy.checkpoint else '0' for strategy in strategy_list])

    config = {
        'pp_deg': pp_size,
        'tp_sizes_enc': tp_sizes_enc,
        'tp_consecutive_flags': tp_consecutive_flags,
        'dp_types_enc': dp_types_enc,
        'use_sp': sp,
        'checkpoint': checkpoint
    }

    return config