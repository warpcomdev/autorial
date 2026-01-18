from __future__ import annotations

import argparse
import configparser
import dataclasses
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel

from modules.config import get_int


@dataclass(frozen=True)
class SpeechConfig:
    """Configuration for Faster-Whisper transcription.

    The module always uses a CUDA device and fails if a GPU is unavailable.
    """

    model_name: str = "large-v3"
    batch_size: int = 16
    beam_size: int = 5
    compute_type: str = "float16"
    language: str | None = "es"


@dataclass(frozen=True)
class SpeechSegment:
    """Represents a transcribed speech segment in seconds."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class OutputDocument:
    """Output document schema for speech transcription results.

    Attributes:
        video: Input video filename.
        segments: Transcribed speech segments.
    """

    video: str
    segments: list[SpeechSegment]


def transcribe_video(
    video_path: Path,
    output_dir: Path,
    config: SpeechConfig,
    vocab_path: Path | None = None,
) -> OutputDocument:
    """Transcribe a video file with Faster-Whisper and write a JSON output.

    Args:
        video_path: Path to the input video file.
        output_dir: Output folder for the generated JSON file.
        config: Configuration for speech transcription.
        vocab_path: Optional path to a vocabulary text file.

    Output:
        Writes ``<stem>.json`` into the output directory. The JSON document
        follows ``OutputDocument``.

    Returns:
        The output document describing the transcription.

    Raises:
        FileNotFoundError: If the input video file does not exist.
        RuntimeError: If ffmpeg is not available or CUDA is unavailable.
    """
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to load audio for transcription")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{video_path.stem}_speech.json"

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Loading speech model %s", config.model_name)
    try:
        model = WhisperModel(
            config.model_name,
            device="cuda",
            compute_type=config.compute_type,
        )
    except Exception as exc:
        raise RuntimeError("CUDA GPU is required for speech transcription") from exc

    batched_model = BatchedInferencePipeline(model=model)
    transcribe_kwargs: dict[str, object] = {
        "batch_size": config.batch_size,
        "beam_size": config.beam_size,
        "language": config.language,
    }
    vocab_prompt = _load_vocab_prompt(vocab_path)
    if vocab_prompt:
        transcribe_kwargs["initial_prompt"] = vocab_prompt
    logging.info("Transcribing %s", video_path.name)
    segments_iter, _info = batched_model.transcribe(str(video_path), **transcribe_kwargs)
    segments: list[SpeechSegment] = []
    for idx, segment in enumerate(segments_iter, start=1):
        segments.append(SpeechSegment(start=segment.start, end=segment.end, text=segment.text))
        if idx % 10 == 0:
            logging.info("Processed %d segments", idx)
    output_doc = OutputDocument(video=video_path.name, segments=segments)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(dataclasses.asdict(output_doc), handle, indent=2, ensure_ascii=False)

    return output_doc


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the speech transcription command.

    Used by the CLI entry point for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe speech from a video file using Faster-Whisper.",
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("output_dir", type=Path, help="Output folder")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to the INI config file.",
    )
    parser.add_argument(
        "-v",
        "--vocab",
        type=Path,
        help="Path to a vocabulary text file (one term per line).",
    )
    return parser.parse_args()


def parse_config(config: configparser.ConfigParser) -> SpeechConfig:
    """Build a SpeechConfig from a loaded configparser instance.

    Input/output paths are provided at runtime; module tuning lives in config.ini.
    """
    defaults = SpeechConfig()
    section = config["speech"] if config.has_section("speech") else {}
    language_value = section.get("language", "").strip()
    language = language_value if language_value else defaults.language
    return SpeechConfig(
        model_name=str(section.get("model_name", defaults.model_name)),
        batch_size=get_int(section, "batch_size", defaults.batch_size),
        beam_size=get_int(section, "beam_size", defaults.beam_size),
        compute_type=str(section.get("compute_type", defaults.compute_type)),
        language=language,
    )


def main(config_path: Path | None = None) -> None:
    """Run speech transcription from the command line.

    This is the CLI entry point used by ``python -m modules.speech``.

    Args:
        config_path: Optional path to the INI config file.
    """
    args = _parse_args()
    parser = configparser.ConfigParser()
    path = args.config or config_path
    if path is not None and path.exists():
        parser.read(path, encoding="utf-8")
    config = parse_config(parser)
    transcribe_video(args.video, args.output_dir, config, args.vocab)


def _load_vocab_prompt(vocab_path: Path | None) -> str:
    """Load vocabulary terms to prime decoding with an initial prompt.

    Used to bias transcription towards domain-specific terms.
    """
    if vocab_path is None:
        return ""
    if not vocab_path.exists():
        raise FileNotFoundError(vocab_path)
    lines = vocab_path.read_text(encoding="utf-8").splitlines()
    terms = [line.strip() for line in lines if line.strip()]
    if not terms:
        return ""
    return "Vocabulario: " + ", ".join(terms)


if __name__ == "__main__":
    main()
