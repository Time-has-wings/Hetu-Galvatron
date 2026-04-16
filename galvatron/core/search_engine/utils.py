import os
import logging

def ensure_log_dir(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def get_thread_logger_single_task(gbsz, chunks, pp_size, global_buffer_tp_size, tp_sp_mode, log_dir='logs'):

    logger_name = f"galvatron_gbsz{gbsz}_chunks{chunks}_pp_size{pp_size}_global_buffer_tp_size{global_buffer_tp_size}_tp_sp_mode{tp_sp_mode}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    log_dir = os.path.join(log_dir, f"search_gbsz{gbsz}_chunks{chunks}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pp{pp_size}_{tp_sp_mode}_buffer_tp{global_buffer_tp_size}.log")
    file_handler = logging.FileHandler(log_file, mode='w')

    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    logger.propagate = False
    
    return logger

def remove_all_galvatron_loggers(prefix='galvatron'):
    manager = logging.Logger.manager
    to_remove = [name for name in manager.loggerDict if name.startswith(prefix)]
    for name in to_remove:
        logger = manager.loggerDict.get(name)
        if isinstance(logger, logging.Logger) and logger.handlers:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        manager.loggerDict.pop(name, None)