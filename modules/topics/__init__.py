"""Public interface for the topics module."""

from .processor import (
    OutputDocument,
    SpeechDocument,
    SpeechSegment,
    TaskItem,
    Topic,
    TopicsConfig,
    generate_topics,
    parse_config,
)

__all__ = [
    "OutputDocument",
    "SpeechDocument",
    "SpeechSegment",
    "TaskItem",
    "Topic",
    "TopicsConfig",
    "generate_topics",
    "parse_config",
]
