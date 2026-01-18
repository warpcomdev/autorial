from __future__ import annotations

import argparse
import configparser
import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ollama import chat
from pydantic import BaseModel, Field, ValidationError

from modules.config import get_float, get_int

PROMPT_TEMPLATE = """You are an expert note-taker analyzing a meeting transcript.

Goal:
- Split the transcript into distinct topical segments ("topics").
- Each topic has a start timestamp; a topic ends when the next topic starts.
- Identify actionable items ("tasks") that occur within topics, each with its own start timestamp.

Guidelines:
- Topics should be concise and descriptive, not overly long.
- Tasks should be imperative or clearly actionable statements.
- Use timestamps from the provided segments as anchors; keep topic/task starts within the segment ranges.
- If there are no clear tasks for a topic, return an empty tasks list.
- Return only JSON that matches the provided schema.

{keywords_block}

Transcript segments:
{segments_block}
"""


@dataclass(frozen=True)
class TopicsConfig:
    """Configuration for topic extraction with Ollama."""

    model_name: str = "gpt-oss:20b"
    temperature: float = 0.0
    context_window: int = 48000
    num_predict: int = 16000


@dataclass(frozen=True)
class SpeechSegment:
    """Represents a transcribed speech segment in seconds."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SpeechDocument:
    """Speech module output document schema.

    Attributes:
        video: Input video filename.
        segments: Transcribed speech segments.
    """

    video: str
    segments: list[SpeechSegment]


@dataclass(frozen=True)
class TaskItem:
    """Represents a task item within a topic."""

    start: float
    task: str


@dataclass(frozen=True)
class Topic:
    """Represents a topic segment."""

    start: float
    topic: str
    tasks: list[TaskItem]


@dataclass(frozen=True)
class OutputDocument:
    """Output document schema for topic extraction results.

    Attributes:
        video: Input video filename.
        topics: Extracted topics with tasks.
    """

    video: str
    topics: list[Topic]


class _TaskModel(BaseModel):
    """Pydantic schema for a task item."""

    start: float = Field(..., description="Task start timestamp in seconds.")
    task: str = Field(..., description="Task description.")


class _TopicModel(BaseModel):
    """Pydantic schema for a topic segment."""

    start: float = Field(..., description="Topic start timestamp in seconds.")
    topic: str = Field(..., description="Topic description.")
    tasks: list[_TaskModel] = Field(default_factory=list)


class _TopicsModel(BaseModel):
    """Pydantic schema for the topics response."""

    topics: list[_TopicModel] = Field(default_factory=list)


def generate_topics(
    speech_json_path: Path,
    output_dir: Path,
    config: TopicsConfig,
    keywords_path: Path | None = None,
) -> OutputDocument:
    """Read speech JSON, call an LLM, and write the topics output JSON.

    Args:
        speech_json_path: Path to the speech module JSON output file.
        output_dir: Output folder for the generated JSON file.
        config: Configuration for topic extraction.
        keywords_path: Optional path to a keywords text file.

    Output:
        Writes ``<stem>_topics.json`` into the output directory. The JSON
        document follows ``OutputDocument``.

    Returns:
        The output document describing topics and tasks.

    Raises:
        FileNotFoundError: If the input JSON file does not exist.
        RuntimeError: If ollama is unavailable or returns invalid output.
        ValueError: If the input JSON schema is invalid.
    """
    if not speech_json_path.exists():
        raise FileNotFoundError(speech_json_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{speech_json_path.stem}_topics.json"

    speech_doc = _load_speech_document(speech_json_path)
    if not speech_doc.segments:
        output_doc = OutputDocument(video=speech_doc.video, topics=[])
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(dataclasses.asdict(output_doc), handle, indent=2)
        return output_doc

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Generating topics with %s", config.model_name)

    prompt = _build_prompt(speech_doc, _load_keywords(keywords_path))
    response = chat(
        model=config.model_name,
        messages=[
            {
                "role": "system",
                "content": "Return the response as JSON and follow the schema exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        format=_TopicsModel.model_json_schema(),
        options={
            "temperature": config.temperature,
            "num_ctx": config.context_window,
            "num_predict": config.num_predict,
        },
    )
    content = response.message.content
    if not content:
        raise RuntimeError("Ollama returned an empty response")

    try:
        topics_model = _TopicsModel.model_validate_json(content)
    except ValidationError as exc:
        raise RuntimeError("Ollama response did not match expected schema") from exc

    output_doc = _model_to_output(speech_doc.video, topics_model)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(dataclasses.asdict(output_doc), handle, indent=2)

    return output_doc


def _load_speech_document(path: Path) -> SpeechDocument:
    """Load the speech JSON document from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    video = data.get("video")
    segments = data.get("segments", [])
    if not isinstance(video, str) or not isinstance(segments, list):
        raise ValueError("Invalid speech JSON format")
    parsed_segments: list[SpeechSegment] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        parsed_segments.append(
            SpeechSegment(
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=str(segment.get("text", "")),
            )
        )
    return SpeechDocument(video=video, segments=parsed_segments)


def _build_prompt(doc: SpeechDocument, keywords: list[str]) -> str:
    """Build the prompt for the LLM from speech segments and keywords."""
    keywords_block = ""
    if keywords:
        keywords_lines = "\n".join(f"- {keyword}" for keyword in keywords)
        keywords_block = f"Keywords (context only):\n{keywords_lines}\n"
    segments_lines = "\n".join(
        f"- [{segment.start:.2f}-{segment.end:.2f}] {segment.text}"
        for segment in doc.segments
    )
    return PROMPT_TEMPLATE.format(
        keywords_block=keywords_block,
        segments_block=segments_lines,
    )


def _model_to_output(video: str, model: _TopicsModel) -> OutputDocument:
    """Convert a parsed pydantic model into output dataclasses."""
    topics: list[Topic] = []
    for topic in model.topics:
        tasks = [TaskItem(start=task.start, task=task.task) for task in topic.tasks]
        topics.append(Topic(start=topic.start, topic=topic.topic, tasks=tasks))
    return OutputDocument(video=video, topics=topics)


def _load_keywords(keywords_path: Path | None) -> list[str]:
    """Load keywords from a text file, one per line."""
    if keywords_path is None:
        return []
    if not keywords_path.exists():
        raise FileNotFoundError(keywords_path)
    lines = keywords_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the topics command.

    Used by the CLI entry point for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Generate topic and task segments from a speech JSON file.",
    )
    parser.add_argument("speech_json", type=Path, help="Speech JSON file")
    parser.add_argument("output_dir", type=Path, help="Output folder")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to the INI config file.",
    )
    parser.add_argument(
        "-k",
        "--keywords",
        type=Path,
        help="Path to a keywords text file (one keyword per line).",
    )
    return parser.parse_args()


def parse_config(config: configparser.ConfigParser) -> TopicsConfig:
    """Build a TopicsConfig from a loaded configparser instance."""
    defaults = TopicsConfig()
    section = config["topics"] if config.has_section("topics") else {}
    model_name = str(section.get("model_name", defaults.model_name))
    return TopicsConfig(
        model_name=model_name,
        temperature=get_float(section, "temperature", defaults.temperature),
        context_window=get_int(section, "context_window", defaults.context_window),
        num_predict=get_int(section, "num_predict", defaults.num_predict),
    )


def main(config_path: Path | None = None) -> None:
    """Run topic extraction from the command line.

    This is the CLI entry point used by ``python -m modules.topics``.

    Args:
        config_path: Optional path to the INI config file.
    """
    args = _parse_args()
    parser = configparser.ConfigParser()
    path = args.config or config_path
    if path is not None and path.exists():
        parser.read(path, encoding="utf-8")
    config = parse_config(parser)
    generate_topics(args.speech_json, args.output_dir, config, args.keywords)


if __name__ == "__main__":
    main()
