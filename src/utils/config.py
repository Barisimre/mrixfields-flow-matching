"""Config loading with shared-network resolution.

A config may declare `network: <file>` (path relative to the config) instead of an
inline `model:` block; the referenced file's keys become `cfg["model"]`. An inline
`model:` block, if present, overrides the shared file key-by-key. Configs with only
an inline `model:` block load unchanged (backward compatible).
"""

from pathlib import Path

import yaml


def load_config(path):
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    net = cfg.get("network")
    if net:
        net_path = Path(net)
        if not net_path.is_absolute():
            net_path = path.parent / net_path
        with open(net_path) as f:
            shared = yaml.safe_load(f)
        cfg["model"] = {**shared, **cfg.get("model", {})}
    return cfg
