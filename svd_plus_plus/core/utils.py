from typing import Any


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
