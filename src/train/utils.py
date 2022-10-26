from typing import Any, Dict, List, MutableMapping, Tuple, Union


def flatten_nested_config(
    config: Union[Dict, MutableMapping],
    parent_key: str = "",
    sep: str = ".",
) -> Dict:
    """Recursively flatten an infinitely nested config. E.g. {"level1":

    {"level2": "level3": {"level4": 5}}}} becomes:

    {"level1.level2.level3.level4": 5}.

    Args:
        d (Union[Dict, MutableMapping]): Dict to flatten.
        parent_key (str): The parent key for the current dict, e.g. "level1" for the
            first iteration. Defaults to "".
        sep (str): How to separate each level in the dict. Defaults to ".".

    Returns:
        Dict: The flattened dict.
    """

    items: List[Tuple[str, Any]] = []
    for k, v in config.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(
                flatten_nested_config(config=v, parent_key=new_key, sep=sep).items()
            )
        else:
            items.append((new_key, v))
    return dict(items)
