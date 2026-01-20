import pandas as pd
from dataclasses import dataclass, fields
from enum import Enum
from .training_utils import ColorSet

def form_strategy(strategy):
    template = '%d-%s-%s'
    assert len(strategy) == 4
    info = strategy[-1]
    pp_deg = strategy[0]
    tp_deg = '%d'%strategy[1]
    dp_deg = '%d'%strategy[2]
    if 'fsdp' in info.keys():
        if info['fsdp']:
            dp_deg += 'f'
    if 'tp' in info.keys():
        if info['tp']:
            tp_deg += '*'
        else:
            dp_deg += '*'
    if 'cpt' in info.keys():
        if info['cpt']:
            dp_deg += '-c'
    
    if 'sp' in info.keys():
        if info['sp']:
            dp_deg += '-sp'
    
    return template%(pp_deg, tp_deg, dp_deg)

def strategy_str2list(strategy_str):
    s = strategy_str.split('-')
    if '*' in s[1]:
        tp_consec = 1
        s[1] = s[1][:-1]
    elif '*' in s[2]:
        tp_consec = 0
        s[2] = s[2][:-1]
    if 'f' in s[2]:
        fsdp = 1
        s[2] = s[2][:-1]
    else:
        fsdp = 0
    cpt = 0
    sp = 0
    if len(s) >= 4:
        if s[3] == 'c':
            cpt = 1
        if s[3] == 'sp':
            sp = 1
    if len(s) >= 5 and s[4] == 'sp':
        sp = 1
    pp_deg, tp_deg, dp_deg = int(s[0]), int(s[1]), int(s[2])
    re = [pp_deg, tp_deg, dp_deg, {}]
    if tp_deg > 1 and dp_deg > 1:
        re[-1]['tp'] = tp_consec
    if dp_deg > 1:
        re[-1]['fsdp'] = fsdp
    if cpt == 1:
        re[-1]['cpt'] = 1
    if sp == 1:
        re[-1]['sp'] = 1
    return re

def print_strategies(strategy_list, logger=None):
    if logger is None:
        if strategy_list is None or isinstance(strategy_list, str):
            print(None)
            return
        if isinstance(strategy_list[0][0],list):
            result_list = []
            for sub_strategy_list in strategy_list:
                sub_result_list = []
                for strategy in sub_strategy_list:
                    sub_result_list.append(form_strategy(strategy))
                result_list.append(', '.join(sub_result_list))
            print(' || '.join(result_list))
        else:
            result_list = []
            for strategy in strategy_list:
                result_list.append(form_strategy(strategy))
            print(', '.join(result_list))
    else:
        if strategy_list is None or isinstance(strategy_list, str):
            logger.info(None)
            return
        if isinstance(strategy_list[0][0],list):
            result_list = []
            for sub_strategy_list in strategy_list:
                sub_result_list = []
                for strategy in sub_strategy_list:
                    sub_result_list.append(form_strategy(strategy))
                result_list.append(', '.join(sub_result_list))
            logger.info(' || '.join(result_list))
        else:
            result_list = []
            for strategy in strategy_list:
                result_list.append(form_strategy(strategy))
            logger.info(', '.join(result_list))

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

DP_TYPE_LIST = DPType.values()

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
    dp_type: DPType = DPType.DDP

    def __post_init__(self):
        self._check_and_fix_dp()
        self._check_tp_sp()

    def _check_and_fix_dp(self):
        if self.dp_size > 1:
            assert DPType.contains(self.dp_type), f"{ColorSet.RED}[ERROR] [{self.__class__.__name__}] When dp_size > 1, strategy.dp_type must be in {DP_TYPE_LIST}, but got '{self.dp_type}'.{ColorSet.RESET}"
        elif self.dp_size == 1 and self.dp_type != DPType.DDP:
            print(f"{ColorSet.YELLOW}[WARNING] [{self.__class__.__name__}] When dp_size is 1, dp_type should be 'DPType.DDP'. Got '{self.dp_type}' instead. Automatically resetting to 'DPType.DDP'.{ColorSet.RESET}")
            self.dp_type = DPType.DDP
    
    def _check_tp_sp(self):
        assert not (self.tp_size > 1 and self.sp_size > 1), f"{ColorSet.RED}[ERROR] [{self.__class__.__name__}] TP and SP cannot be used together. Got tp_size={self.tp_size} and sp_size={self.sp_size}.{ColorSet.RESET}"

    @property
    def world_size(self):
        return self.pp_size * self.tp_size * self.sp_size * self.cp_size * self.dp_size

    @property
    def sdp_size(self):
        return self.dp_size * self.sp_size

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
    dp_type: DPType = DPType.DDP
    checkpoint: bool = False

    def __post_init__(self):
        self._check_and_fix_dp()

    def _check_and_fix_dp(self):
        if self.dp_size > 1:
            assert DPType.contains(self.dp_type), f"{ColorSet.RED}[ERROR] [{self.__class__.__name__}] When dp_size > 1, strategy.dp_type must be in {DP_TYPE_LIST}, but got '{self.dp_type}'.{ColorSet.RESET}"
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

byte_to_MB = 1024 * 1024
model_states_to_param_size_ratio = 4