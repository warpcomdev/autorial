from __future__ import annotations

import argparse
import configparser
import dataclasses
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ollama import chat

from modules.config import get_float, get_int

PROMPT_TEMPLATE = """You are writing a training-style tutorial in markdown.

Instructions:
- Use the section material below to write a clear, step-by-step guide.
- Reference screenshots from the img folder using their hash names, e.g. ![](img/<hash>.jpg).
- You do not need to use every selected image; only include screenshots that add value to the documentation.
- The selections come from a small model and may be inaccurate; prioritize tasks and speech over selections.
- Speech fragments may contain typos or misheard terms; correct them when needed.
- Keep the output strictly in markdown.
- Write the markdown in this language: {language}.

Previous section summary:
{previous_summary}

Section title:
{section_title}

Section tasks and context:
{section_context}
"""

SUMMARY_TEMPLATE = """Summarize the following markdown section in a single paragraph.

Markdown:
{markdown}
"""


@dataclass(frozen=True)
class MarkdownConfig:
    """Configuration for markdown generation with an LLM."""

    model_name: str = "gpt-oss:20b"
    temperature: float = 0.0
    context_window: int = 32000
    num_predict: int = 32000
    summary_num_predict: int = 1024
    language: str = "es"
    image_kind: str = "original"
    image_format: str = "jpg"


@dataclass(frozen=True)
class SpeechSegment:
    """Represents a transcribed speech segment in seconds."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class CombinedTask:
    """Represents a task from the combine module."""

    start: float
    task: str
    speech_segments: list[SpeechSegment]
    keyframes: list[str]


@dataclass(frozen=True)
class CombinedTopic:
    """Represents a topic from the combine module."""

    start: float
    topic: str
    tasks: list[CombinedTask]


@dataclass(frozen=True)
class CombineDocument:
    """Combine module output document schema."""

    video: str
    topics: list[CombinedTopic]


@dataclass(frozen=True)
class SelectionItem:
    """Represents a selected image for a task."""

    hash: str
    description: str
    reason: str


@dataclass(frozen=True)
class SelectionTask:
    """Represents selections for a task."""

    start: float
    task: str
    selections: list[SelectionItem]


@dataclass(frozen=True)
class SelectionTopic:
    """Represents selections for a topic."""

    start: float
    topic: str
    tasks: list[SelectionTask]


@dataclass(frozen=True)
class SelectionDocument:
    """Selection module output document schema."""

    video: str
    topics: list[SelectionTopic]


@dataclass(frozen=True)
class SectionSummary:
    """Summary entry for a markdown section."""

    index: int
    title: str
    summary: str
    markdown_file: str


@dataclass(frozen=True)
class MarkdownOutput:
    """Output summary for markdown generation."""

    video: str
    sections: list[SectionSummary]


def generate_markdown(
    combine_json_path: Path,
    selection_json_path: Path,
    output_dir: Path,
    config: MarkdownConfig,
) -> MarkdownOutput:
    """Generate markdown sections from combine and selection outputs.

    Args:
        combine_json_path: Path to the combine module JSON output.
        selection_json_path: Path to the selection module JSON output.
        output_dir: Output folder containing the per-video directory.
        config: Configuration for markdown generation.

    Output:
        Writes markdown files into ``<output_dir>/<video_stem>/`` and a README.md
        with section summaries. Selected screenshots are stored in
        ``<output_dir>/<video_stem>/img``.

    Returns:
        A summary object describing generated sections.

    Raises:
        FileNotFoundError: If input JSON or the hash database is missing.
        ValueError: If the combine and selection documents do not align.
        RuntimeError: If the LLM returns empty responses.
    """
    if not combine_json_path.exists():
        raise FileNotFoundError(combine_json_path)
    if not selection_json_path.exists():
        raise FileNotFoundError(selection_json_path)

    combine_doc = _load_combine_document(combine_json_path)
    selection_doc = _load_selection_document(selection_json_path)
    _validate_alignment(combine_doc, selection_doc)

    output_dir.mkdir(parents=True, exist_ok=True)
    video_stem = Path(combine_doc.video).stem if combine_doc.video else combine_json_path.stem
    video_dir = output_dir / video_stem
    video_dir.mkdir(parents=True, exist_ok=True)
    img_dir = video_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    db_path = video_dir / "frames.sqlite3"
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Generating markdown with %s", config.model_name)

    with sqlite3.connect(db_path) as connection:
        _extract_selected_images(connection, selection_doc, img_dir, config)

    summaries: list[SectionSummary] = []
    previous_summary = ""
    total_topics = len(combine_doc.topics)
    for index, (topic, selection_topic) in enumerate(
        zip(combine_doc.topics, selection_doc.topics, strict=True),
        start=1,
    ):
        logging.info("Section %d/%d: %s", index, total_topics, topic.topic)
        section_context = _build_section_context(topic, selection_topic)
        prompt = PROMPT_TEMPLATE.format(
            previous_summary=previous_summary or "None.",
            section_title=topic.topic,
            section_context=section_context,
            language=config.language,
        )
        markdown = _call_llm(prompt, config, num_predict=config.num_predict)
        section_filename = f"section_{index:04d}.md"
        section_path = video_dir / section_filename
        section_path.write_text(markdown, encoding="utf-8")

        summary_prompt = SUMMARY_TEMPLATE.format(markdown=markdown)
        summary = _call_llm(summary_prompt, config, num_predict=config.summary_num_predict)
        logging.info("Summary %d: %s", index, summary)
        summaries.append(
            SectionSummary(
                index=index,
                title=topic.topic,
                summary=summary,
                markdown_file=section_filename,
            )
        )
        previous_summary = summary

    _write_readme(video_dir, summaries)
    return MarkdownOutput(video=combine_doc.video, sections=summaries)


def _call_llm(prompt: str, config: MarkdownConfig, num_predict: int) -> str:
    """Call the LLM with the provided prompt."""
    response = chat(
        model=config.model_name,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": config.temperature,
            "num_ctx": config.context_window,
            "num_predict": num_predict,
        },
    )
    content = response.message.content if response else ""
    if not content:
        raise RuntimeError("Ollama returned an empty response")
    return content.strip()


def _extract_selected_images(
    connection: sqlite3.Connection,
    selection_doc: SelectionDocument,
    img_dir: Path,
    config: MarkdownConfig,
) -> None:
    """Extract original images for selected hashes into the img folder."""
    hashes = _collect_selection_hashes(selection_doc)
    for hash_value in hashes:
        output_path = img_dir / f"{hash_value}.{config.image_format}"
        if output_path.exists():
            continue
        blob = _load_frame_blob(connection, hash_value, config.image_kind)
        if blob is None:
            continue
        output_path.write_bytes(blob)


def _collect_selection_hashes(selection_doc: SelectionDocument) -> set[str]:
    """Collect unique selection hashes from the selection document."""
    hashes: set[str] = set()
    for topic in selection_doc.topics:
        for task in topic.tasks:
            for selection in task.selections:
                hashes.add(selection.hash)
    return hashes


def _load_frame_blob(
    connection: sqlite3.Connection,
    hash_value: str,
    kind: str,
) -> bytes | None:
    """Load a frame blob from SQLite by hash and kind."""
    cursor = connection.execute(
        "SELECT data FROM frames WHERE hash = ? AND kind = ?",
        (hash_value, kind),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return row[0]


def _build_section_context(topic: CombinedTopic, selection_topic: SelectionTopic) -> str:
    """Build the section context text for the LLM prompt."""
    lines: list[str] = []
    for task, selection_task in zip(topic.tasks, selection_topic.tasks, strict=True):
        lines.append(f"Task: {task.task}")
        if task.speech_segments:
            lines.append("Speech fragments:")
            for segment in task.speech_segments:
                lines.append(f"- [{segment.start:.2f}-{segment.end:.2f}] {segment.text}")
        if selection_task.selections:
            lines.append("Selected screenshots:")
            for selection in selection_task.selections:
                lines.append(f"- {selection.hash}: {selection.description} ({selection.reason})")
        lines.append("")
    return "\n".join(lines).strip()


def _write_readme(video_dir: Path, summaries: list[SectionSummary]) -> None:
    """Write the README.md with section summaries."""
    lines = ["# Tutorial Sections", ""]
    for summary in summaries:
        lines.append(f"- [{summary.title}]({summary.markdown_file})")
        lines.append(f"  - {summary.summary}")
    (video_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _load_combine_document(path: Path) -> CombineDocument:
    """Load combine JSON document from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    video = data.get("video")
    topics = data.get("topics", [])
    if not isinstance(video, str) or not isinstance(topics, list):
        raise ValueError("Invalid combine JSON format")
    parsed_topics: list[CombinedTopic] = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        tasks = topic.get("tasks", [])
        parsed_tasks: list[CombinedTask] = []
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                segments = task.get("speech_segments", [])
                parsed_segments: list[SpeechSegment] = []
                if isinstance(segments, list):
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
                parsed_tasks.append(
                    CombinedTask(
                        start=float(task.get("start", 0.0)),
                        task=str(task.get("task", "")),
                        speech_segments=parsed_segments,
                        keyframes=[str(value) for value in task.get("keyframes", [])],
                    ),
                )
        parsed_topics.append(
            CombinedTopic(
                start=float(topic.get("start", 0.0)),
                topic=str(topic.get("topic", "")),
                tasks=parsed_tasks,
            ),
        )
    return CombineDocument(video=video, topics=parsed_topics)


def _load_selection_document(path: Path) -> SelectionDocument:
    """Load selection JSON document from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    video = data.get("video")
    topics = data.get("topics", [])
    if not isinstance(video, str) or not isinstance(topics, list):
        raise ValueError("Invalid selection JSON format")
    parsed_topics: list[SelectionTopic] = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        tasks = topic.get("tasks", [])
        parsed_tasks: list[SelectionTask] = []
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                selections = task.get("selections", [])
                parsed_selections: list[SelectionItem] = []
                if isinstance(selections, list):
                    for selection in selections:
                        if not isinstance(selection, dict):
                            continue
                        parsed_selections.append(
                            SelectionItem(
                                hash=str(selection.get("hash", "")),
                                description=str(selection.get("description", "")),
                                reason=str(selection.get("reason", "")),
                            )
                        )
                parsed_tasks.append(
                    SelectionTask(
                        start=float(task.get("start", 0.0)),
                        task=str(task.get("task", "")),
                        selections=parsed_selections,
                    ),
                )
        parsed_topics.append(
            SelectionTopic(
                start=float(topic.get("start", 0.0)),
                topic=str(topic.get("topic", "")),
                tasks=parsed_tasks,
            ),
        )
    return SelectionDocument(video=video, topics=parsed_topics)


def _validate_alignment(combine: CombineDocument, selection: SelectionDocument) -> None:
    """Ensure combine and selection documents are aligned."""
    if len(combine.topics) != len(selection.topics):
        raise ValueError("Combine and selection topic counts do not match")
    for topic, selection_topic in zip(combine.topics, selection.topics, strict=True):
        if len(topic.tasks) != len(selection_topic.tasks):
            raise ValueError("Combine and selection task counts do not match")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the markdown command.

    Used by the CLI entry point for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Generate markdown tutorials from combined outputs.",
    )
    parser.add_argument("combine_json", type=Path, help="Combine JSON file")
    parser.add_argument("selection_json", type=Path, help="Selection JSON file")
    parser.add_argument("output_dir", type=Path, help="Output folder")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to the INI config file.",
    )
    return parser.parse_args()


def parse_config(config: configparser.ConfigParser) -> MarkdownConfig:
    """Build a MarkdownConfig from a loaded configparser instance."""
    defaults = MarkdownConfig()
    section = config["markdown"] if config.has_section("markdown") else {}
    return MarkdownConfig(
        model_name=str(section.get("model_name", defaults.model_name)),
        temperature=get_float(section, "temperature", defaults.temperature),
        context_window=get_int(section, "context_window", defaults.context_window),
        num_predict=get_int(section, "num_predict", defaults.num_predict),
        summary_num_predict=get_int(section, "summary_num_predict", defaults.summary_num_predict),
        language=str(section.get("language", defaults.language)),
        image_kind=str(section.get("image_kind", defaults.image_kind)),
        image_format=str(section.get("image_format", defaults.image_format)),
    )


def main(config_path: Path | None = None) -> None:
    """Run markdown generation from the command line.

    This is the CLI entry point used by ``python -m modules.markdown``.

    Args:
        config_path: Optional path to the INI config file.
    """
    args = _parse_args()
    parser = configparser.ConfigParser()
    path = args.config or config_path
    if path is not None and path.exists():
        parser.read(path, encoding="utf-8")
    config = parse_config(parser)
    generate_markdown(args.combine_json, args.selection_json, args.output_dir, config)


if __name__ == "__main__":
    main()
