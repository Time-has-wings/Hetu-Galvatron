import os
import logging

def ensure_log_dir(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)  # Thread-safe: exist_ok=True prevents race condition
    return log_dir

def get_thread_logger(gbsz, chunks, pp_size, tp_sp_mode,tp_size_limit, log_dir=None):
    assert log_dir is not None

    logger_name = f"galvatron-bsz{gbsz}-chunks{chunks}-pp_size{pp_size}-tp_sp_mode{tp_sp_mode}-tp_size_limit{tp_size_limit}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    log_dir = os.path.join(log_dir, f"search_bsz{gbsz}_chunk{chunks}")
    os.makedirs(log_dir, exist_ok=True)  # Thread-safe: exist_ok=True prevents race condition
    log_file = os.path.join(log_dir, f"pp_size{pp_size}-{tp_sp_mode}-tp_size_limit{tp_size_limit}.log")
    file_handler = logging.FileHandler(log_file, mode='w')

    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    logger.propagate = False
    
    return logger

# def get_thread_logger_version3(gbsz, chunks, pp_size, tp_sp_mode, tp_size_limit, log_dir=None):
#     assert log_dir is not None
    
#     logger_name = f"galvatron-gbsz{gbsz}-chunk{chunks}-pp_size{pp_size}-{tp_sp_mode}-tp_size_limit{tp_size_limit}"
#     logger = logging.getLogger(logger_name)
#     if logger.handlers:
#         return logger
#     logger.setLevel(logging.INFO)
#     log_dir = os.path.join(log_dir, f"galvatron-gbsz{gbsz}-chunk{chunks}")
#     ensure_log_dir(log_dir)
#     logger_path = os.path.join(log_dir, f"pp_size{pp_size}-{tp_sp_mode}-tp_size_limit{tp_size_limit}.log")
#     file_handler = logging.FileHandler(logger_path, mode='w')
#     formatter = logging.Formatter('%(message)s')
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
#     logger.propagate = False
#     return logger

# def get_thread_logger_version3(gbsz, chunks, pp_size, extra_name, log_dir='logs'):
#     logger_name = f"galvatron-gbsz{gbsz}-chunk{chunks}-pp_size{pp_size}"
#     for key, value in extra_name.items():
#         logger_name += f'-{key}{value}'

#     logger = logging.getLogger(logger_name)

#     if logger.handlers:
#         return logger
        
#     logger.setLevel(logging.INFO)
    