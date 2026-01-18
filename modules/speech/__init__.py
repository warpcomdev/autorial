"""Public interface for the speech transcription module."""

from .processor import OutputDocument, SpeechConfig, SpeechSegment, parse_config, transcribe_video

__all__ = [
    "OutputDocument",
    "SpeechConfig",
    "SpeechSegment",
    "parse_config",
    "transcribe_video",
]
