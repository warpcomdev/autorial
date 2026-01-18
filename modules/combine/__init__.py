"""Public interface for the combine module."""

from .processor import CombineConfig, CombinedDocument, CombinedTask, CombinedTopic, combine_outputs, parse_config

__all__ = [
    "CombineConfig",
    "CombinedDocument",
    "CombinedTask",
    "CombinedTopic",
    "combine_outputs",
    "parse_config",
]
