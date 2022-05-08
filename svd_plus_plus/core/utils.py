from typing import Any
import os
import random

import numpy as np
import torch


def make_flat(config: dict[str, Any]) -> dict[str, Any]:
    """
    Returns the parameters of a flat dictionary from keys to values.
    Nested structure is collapsed with periods.
    """
    flat_params = {}

    def recurse(parameters: dict[str, Any], path: list[str]) -> None:
        for key, value in parameters.items():
            new_path = path + [key]
            if isinstance(value, dict):
                recurse(value, new_path)
            else:
                flat_params[".".join(new_path)] = value

    recurse(config, [])
    return flat_params


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
