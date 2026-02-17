"""Utility functions for Goodseed."""

import random
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

# Word lists for readable run IDs (adjective-animal format)
ADJECTIVES = [
    "agile", "bold", "calm", "daring", "eager", "fierce", "gentle", "happy",
    "idle", "jolly", "keen", "lively", "merry", "noble", "proud", "quick",
    "rapid", "serene", "swift", "tame", "unique", "vivid", "warm", "zealous",
    "amber", "azure", "coral", "dusty", "ebony", "frosty", "golden", "hazy",
    "ivory", "jade", "khaki", "lunar", "misty", "navy", "olive", "pearl",
]

ANIMALS = [
    "albatross", "badger", "caracal", "dolphin", "eagle", "falcon", "gazelle",
    "heron", "ibis", "jaguar", "koala", "lemur", "meerkat", "narwhal", "otter",
    "panther", "quokka", "raven", "salmon", "toucan", "urchin", "viper", "walrus",
    "barracuda", "cheetah", "dragonfly", "flamingo", "giraffe", "hedgehog",
    "iguana", "jellyfish", "kestrel", "leopard", "mantis", "newt", "octopus",
    "pelican", "quail", "rhino", "starfish", "turtle", "vulture", "wombat",
]


def generate_run_name() -> str:
    """Generate a run name like 'bold-falcon'."""
    adj = random.choice(ADJECTIVES)
    animal = random.choice(ANIMALS)
    return f"{adj}-{animal}"


# Supported value types for logging
SupportedValue = Union[bool, int, float, str, datetime, None]


def is_supported_type(value: Any) -> bool:
    """Check if a value is a directly supported type."""
    return isinstance(value, (bool, int, float, str, datetime, type(None)))


def cast_to_string(value: Any) -> str:
    """Cast an unsupported value to string representation."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def serialize_value(value: Any) -> Tuple[str, str]:
    """Serialize a value for storage, returning (type_tag, serialized_string).

    All values are serialized as strings for SQLite TEXT columns.
    """
    if value is None:
        return ("null", "")
    elif isinstance(value, bool):
        return ("bool", "true" if value else "false")
    elif isinstance(value, int):
        return ("int", str(value))
    elif isinstance(value, float):
        return ("float", str(value))
    elif isinstance(value, str):
        return ("str", value)
    elif isinstance(value, datetime):
        return ("datetime", value.isoformat())
    else:
        return ("str", str(value))


def deserialize_value(type_tag: str, raw_value: Any) -> Any:
    """Deserialize a value from storage."""
    if type_tag == "null" or raw_value is None:
        return None
    elif type_tag == "bool":
        if isinstance(raw_value, str):
            return raw_value.lower() == "true"
        return bool(raw_value)
    elif type_tag == "int":
        return int(raw_value)
    elif type_tag == "float":
        return float(raw_value)
    elif type_tag == "str":
        return str(raw_value)
    elif type_tag == "datetime":
        return datetime.fromisoformat(raw_value)
    else:
        return raw_value


def flatten_dict(
    data: Dict[str, Any],
    parent_key: str = "",
    sep: str = "/",
    cast_unsupported: bool = False,
) -> Dict[str, SupportedValue]:
    """Flatten a nested dictionary into path-based keys.

    Args:
        data: The nested dictionary to flatten.
        parent_key: Prefix for all keys (used in recursion).
        sep: Separator for nested keys.
        cast_unsupported: If True, cast unsupported types to strings.

    Returns:
        A flat dictionary with path-based keys.

    Raises:
        TypeError: If a value is unsupported and cast_unsupported is False.
    """
    items: List[Tuple[str, SupportedValue]] = []

    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            items.extend(
                flatten_dict(value, new_key, sep, cast_unsupported).items()
            )
        elif isinstance(value, (list, tuple)):
            # Convert lists to indexed keys
            for i, item in enumerate(value):
                item_key = f"{new_key}/{i}"
                if isinstance(item, dict):
                    items.extend(
                        flatten_dict(item, item_key, sep, cast_unsupported).items()
                    )
                elif is_supported_type(item):
                    items.append((item_key, item))
                elif cast_unsupported:
                    items.append((item_key, cast_to_string(item)))
                else:
                    raise TypeError(
                        f"Unsupported type {type(item).__name__} at {item_key}"
                    )
        elif is_supported_type(value):
            items.append((new_key, value))
        elif cast_unsupported:
            items.append((new_key, cast_to_string(value)))
        else:
            raise TypeError(
                f"Unsupported type {type(value).__name__} at {new_key}"
            )

    return dict(items)


def normalize_path(path: str) -> str:
    """Normalize a metric/config path (strip leading/trailing slashes)."""
    return path.strip("/")

