"""Public interface for the selection module."""

from .processor import (
    CombineDocument,
    CombinedTask,
    CombinedTopic,
    ImageSelection,
    SelectionConfig,
    SelectionDocument,
    SelectionTask,
    SelectionTopic,
    select_keyframes,
    parse_config,
)

__all__ = [
    "CombineDocument",
    "CombinedTask",
    "CombinedTopic",
    "ImageSelection",
    "SelectionConfig",
    "SelectionDocument",
    "SelectionTask",
    "SelectionTopic",
    "select_keyframes",
    "parse_config",
]
