import pandas as pd
from dataclasses import dataclass, fields
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

@dataclass
class GalvatronStrategy:
    pp_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    tp_of_ep_size: int = 1
    use_ulysses: bool = False
    dp_type: str = 'zero0' # ['zero0', 'zero2', 'zero3']
    checkpoint: bool = False
    
    unit: str = 'all' # ['all'. 'attention', 'ffn', 'embedding_lmhead']
    is_moe: bool = False
    
    def __post_init__(self):        
        # Check dp_type
        if self.dp_size > 1:
            assert self.dp_type in ["zero0", "zero2", "zero3"], f"{ColorSet.RED}[ERROR] [{self.__class__.__name__}] When dp_size > 1, strategy.dp_type must be in ['zero0', 'zero2', 'zero3'], but got '{self.dp_type}'.{ColorSet.RESET}"
        elif self.dp_size == 1 and self.dp_type != 'zero0':
            print(f"{ColorSet.YELLOW}[WARNING] [{self.__class__.__name__}] When dp_size is 1, dp_type should be 'zero0'. Got '{self.dp_type}' instead. Automatically resetting to 'zero0'.{ColorSet.RESET}")
            self.dp_type = 'zero0'
        
        # Check unit
        assert self.unit in ['all', 'attention', 'ffn', 'embedding_lmhead'], f"{ColorSet.RED}[ERROR] [{self.__class__.__name__}] Invalid strategy.unit value: '{self.unit}'. Must be one of ['all', 'attention', 'ffn', 'embedding_lmhead'].{ColorSet.RESET}"
        
        # Modify use_ulysses
        if self.is_moe and self.tp_of_ep_size == 1 and self.use_ulysses:
            self.use_ulysses = False
        elif self.is_moe == False and self.tp_size == 1 and self.use_ulysses:
            self.use_ulysses = False

        # Modify Checkpoint
        if self.unit == 'embedding_lmhead':
            self.checkpoint = False
        
        # Maintain certain values based on whether the model is a MoE model.
        # if self.is_moe == False:
        #     self.ep_size = 1
        #     self.tp_of_ep_size = 1
        # else:
        #     self.dp_size = 1
        #     self.tp_size = 1
        #     self.cp_size = 1
        #     self.use_ulysses = False
        #     self.dp_type = 'zero0'

    @property
    def world_size(self):
        if self.unit in ['all', 'attention']:
            return self.pp_size * self.tp_size * self.dp_size * self.cp_size
        elif self.unit == 'ffn':
            return self.pp_size * self.ep_size * self.tp_of_ep_size
        elif self.unit == 'embedding_lmhead':
            return self.pp_size * self.tp_size * self.dp_size * self.cp_size
        else:
            raise ValueError(f'unit should in ["all", "attention", "mlp"], but got {self.unit}')
    
    def __eq__(self, other):
        if not isinstance(other, GalvatronStrategy):
            return False
        for field in self.__dataclass_fields__:
            if getattr(self, field) != getattr(other, field):
                return False
        return True
    
    def __lt__(self, other):
        if not isinstance(other, GalvatronStrategy):
            return NotImplemented
        
        for field in self.__dataclass_fields__:
            self_val = getattr(self, field)
            other_val = getattr(other, field)
            
            if self_val < other_val:
                return True
            elif self_val > other_val:
                return False
        
        return False
    
    def __hash__(self):
        attrs = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        return hash(attrs)
    
    def __str__(self):
        return f"Strategy({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"
    
def strategy_list_to_csv(strategy_list, csv_save_file=None):
    field_names = [field.name for field in fields(GalvatronStrategy)]
    data = [{f: getattr(s, f) for f in field_names} for s in strategy_list]
    df = pd.DataFrame(data)
    
    if csv_save_file is not None:
        df.to_csv(csv_save_file, index=True)
        print(f"\nExported to CSV file: {csv_save_file}")
    
    return df.to_string(index=True)

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0