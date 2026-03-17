from pathlib import Path
from typing import Any, List, Optional

from galvatron.core.args_schema import CoreArgs
from omegaconf import OmegaConf


def load_with_hydra(
    config_path: str,
    overrides: Optional[List[str]] = None,
    mode: Optional[str] = None,
    **hydra_kwargs: Any,
) -> CoreArgs:
    from hydra import compose, initialize_config_dir
    path = Path(config_path).resolve()
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        cfg = compose(config_name=path.name, overrides=overrides or [], **hydra_kwargs)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    args = CoreArgs(**config_dict)
    if mode == "train_dist":
        args = args.runtime
    elif mode == "profile":
        args = args.profiler
    elif mode == "search":
        args = args.search_engine
    return args