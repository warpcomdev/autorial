from __future__ import annotations

import argparse
import configparser
import dataclasses
import json
import logging
import sqlite3
import multiprocessing
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ollama import chat
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, ValidationError

from modules.config import get_float, get_int

PROMPT_TEMPLATE = """You are a documentation assistant working from a transcript.

Goal:
- Select the most relevant screenshot for each distinct action described.
- A task can involve multiple actions; choose exactly one image per action.
- For each selection, describe what the screenshot shows and why it is relevant.

Guidelines:
- Prefer images that clearly show the UI state where the action happens.
- If two images are similar, pick the one that better captures the action.
- Use the transcript text as the primary signal for what the action is.
- If no image is relevant to the task, return an empty selections list.
- Each image has a visible index watermark in the corner; return only those indices.
- Return only JSON that matches the provided schema.

Topic:
{topic}

Task:
{task}

Transcript context:
{context_block}

Candidate images (by index):
{images_block}
"""


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for keyframe selection with a multimodal model."""

    model_name: str = "qwen3-vl:8b"
    temperature: float = 0.0
    context_window: int = 64000
    num_predict: int = 16000
    image_kind: str = "resized"
    image_format: str = "jpg"
    max_retries: int = 2
    pool_size: int = 4
    request_delay: float = 0.2
    retry_backoff: float = 0.5


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
class ImageSelection:
    """Represents a selected image with description and relevance."""

    hash: str
    description: str
    reason: str


@dataclass(frozen=True)
class SelectionTask:
    """Represents a task with selected images."""

    start: float
    task: str
    selections: list[ImageSelection]


@dataclass(frozen=True)
class SelectionTopic:
    """Represents a topic with selected task images."""

    start: float
    topic: str
    tasks: list[SelectionTask]


@dataclass(frozen=True)
class SelectionDocument:
    """Selection output document schema."""

    video: str
    topics: list[SelectionTopic]


class _SelectionItem(BaseModel):
    """Pydantic schema for a single image selection."""

    index: int = Field(..., description="Index of the selected image.")
    description: str = Field(..., description="Description of the screenshot.")
    reason: str = Field(..., description="Why the screenshot is relevant to the task.")


class _SelectionResponse(BaseModel):
    """Pydantic schema for the selection response."""

    selections: list[_SelectionItem] = Field(default_factory=list)


@dataclass(frozen=True)
class _TaskPayload:
    """Payload for selecting images for a single task."""

    topic_index: int
    task_index: int
    topic: CombinedTopic
    task: CombinedTask


@dataclass(frozen=True)
class _TaskResult:
    """Result for a selection task."""

    topic_index: int
    task_index: int
    topic_title: str
    task_title: str
    selections: list[ImageSelection]


def select_keyframes(
    combine_json_path: Path,
    output_dir: Path,
    config: SelectionConfig,
) -> SelectionDocument:
    """Select relevant keyframes for each task using a multimodal model.

    Args:
        combine_json_path: Path to the combine module JSON output file.
        output_dir: Output folder for the generated JSON file.
        config: Configuration for keyframe selection.

    Output:
        Writes ``<stem>_selection.json`` into the output directory. The JSON
        document follows ``SelectionDocument``.

    Returns:
        The selection document with chosen images and captions.

    Raises:
        FileNotFoundError: If the input JSON or SQLite database does not exist.
        RuntimeError: If ollama returns invalid output or is unavailable.
        ValueError: If the input JSON schema is invalid.
    """
    if not combine_json_path.exists():
        raise FileNotFoundError(combine_json_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{combine_json_path.stem}_selection.json"

    combine_doc = _load_combine_document(combine_json_path)
    db_path = _resolve_db_path(combine_json_path, combine_doc.video)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Selecting keyframes with %s", config.model_name)

    task_payloads: list[_TaskPayload] = []
    for topic_index, topic in enumerate(combine_doc.topics, start=1):
        for task_index, task in enumerate(topic.tasks, start=1):
            task_payloads.append(
                _TaskPayload(
                    topic_index=topic_index,
                    task_index=task_index,
                    topic=topic,
                    task=task,
                ),
            )

    results: dict[tuple[int, int], _TaskResult] = {}
    if task_payloads:
        if config.pool_size > 1:
            with multiprocessing.Pool(processes=config.pool_size) as pool:
                for result in pool.imap_unordered(
                    _select_task_worker,
                    [(db_path, payload, config) for payload in task_payloads],
                ):
                    results[(result.topic_index, result.task_index)] = result
        else:
            for payload in task_payloads:
                result = _select_task_worker((db_path, payload, config))
                results[(result.topic_index, result.task_index)] = result

    selection_topics: list[SelectionTopic] = []
    total_topics = len(combine_doc.topics)
    for topic_index, topic in enumerate(combine_doc.topics, start=1):
        logging.info("Topic %d/%d: %s", topic_index, total_topics, topic.topic)
        selected_tasks: list[SelectionTask] = []
        for task_index, task in enumerate(topic.tasks, start=1):
            result = results.get((topic_index, task_index))
            selections = result.selections if result else []
            selected_tasks.append(
                SelectionTask(start=task.start, task=task.task, selections=selections),
            )
        selection_topics.append(
            SelectionTopic(start=topic.start, topic=topic.topic, tasks=selected_tasks),
        )

    output_doc = SelectionDocument(video=combine_doc.video, topics=selection_topics)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(dataclasses.asdict(output_doc), handle, indent=2)

    return output_doc


def _select_for_task(
    db_path: Path,
    topic_index: int,
    task_index: int,
    topic: CombinedTopic,
    task: CombinedTask,
    config: SelectionConfig,
) -> list[ImageSelection]:
    """Select relevant images for a single task."""
    if not task.keyframes:
        return []

    hash_candidates = list(task.keyframes)
    if len(hash_candidates) > 32:
        hash_candidates = _filter_similar_hashes(hash_candidates)

    with sqlite3.connect(db_path) as connection, tempfile.TemporaryDirectory() as temp_root:
        temp_dir = Path(temp_root)
        image_paths: list[Path] = []
        index_to_hash: list[str] = []
        for index, hash_value in enumerate(hash_candidates):
            path = _extract_frame_blob(connection, hash_value, config, temp_dir)
            if path is None:
                continue
            _watermark_image(path, index)
            index_to_hash.append(hash_value)
            image_paths.append(path)

        if not image_paths:
            return []

        prompt = _build_prompt(topic, task, index_to_hash)
        if config.request_delay > 0:
            time.sleep(config.request_delay)
        content = ""
        invalid_content: str | None = None
        for attempt in range(1, config.max_retries + 2):
            response = chat(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt, "images": [str(p) for p in image_paths]}],
                format=_SelectionResponse.model_json_schema(),
                options={
                    "temperature": config.temperature,
                    "num_ctx": config.context_window,
                    "num_predict": config.num_predict,
                },
            )
            content = response.message.content or ""
            if content.strip():
                break
            logging.warning("Empty response from Ollama (attempt %d)", attempt)
            if attempt <= config.max_retries and config.retry_backoff > 0:
                time.sleep(config.retry_backoff * attempt)
            if not content.strip():
                logging.info("  Prompt:\n%s", prompt)
                logging.warning("Skipping task due to empty response: %s", task.task)
                return []

        try:
            selection_response = _SelectionResponse.model_validate_json(content)
        except ValidationError:
            logging.warning("Invalid JSON response from Ollama (attempt %d)", attempt)
            invalid_content = content
            if attempt <= config.max_retries and config.retry_backoff > 0:
                time.sleep(config.retry_backoff * attempt)
            content = ""
            continue

        selections: list[ImageSelection] = []
        for item in selection_response.selections:
            if item.index < 0 or item.index >= len(index_to_hash):
                continue
            selections.append(
                ImageSelection(
                    hash=index_to_hash[item.index],
                    description=item.description,
                    reason=item.reason,
                ),
            )
        return selections
    if invalid_content:
        _save_invalid_response(
            db_path,
            topic_index,
            task_index,
            task.task,
            invalid_content,
        )
    logging.info("  Prompt:\n%s", prompt)
    logging.warning("Skipping task due to invalid responses: %s", task.task)
    return []


def _select_task_worker(args: tuple[Path, _TaskPayload, SelectionConfig]) -> _TaskResult:
    """Worker to select images for a single task."""
    db_path, payload, config = args
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(
        "  Starting topic %d task %d: %s",
        payload.topic_index,
        payload.task_index,
        payload.task.task,
    )
    selections = _select_for_task(
        db_path,
        payload.topic_index,
        payload.task_index,
        payload.topic,
        payload.task,
        config,
    )
    logging.info(
        "  Completed topic %d task %d: %s (%d selections)",
        payload.topic_index,
        payload.task_index,
        payload.task.task,
        len(selections),
    )
    for selection in selections:
        logging.info(
            "    Selection: %s | %s",
            selection.description,
            selection.reason,
        )
    return _TaskResult(
        topic_index=payload.topic_index,
        task_index=payload.task_index,
        topic_title=payload.topic.topic,
        task_title=payload.task.task,
        selections=selections,
    )


def _extract_frame_blob(
    connection: sqlite3.Connection,
    hash_value: str,
    config: SelectionConfig,
    temp_dir: Path,
) -> Path | None:
    """Extract a frame blob from SQLite and write it to a temp file."""
    cursor = connection.execute(
        "SELECT data FROM frames WHERE hash = ? AND kind = ?",
        (hash_value, config.image_kind),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    data = row[0]
    path = temp_dir / f"{hash_value}.{config.image_format}"
    path.write_bytes(data)
    return path


def _save_invalid_response(
    db_path: Path,
    topic_index: int,
    task_index: int,
    task_title: str,
    content: str,
) -> None:
    """Persist invalid LLM output for debugging."""
    errors_dir = db_path.parent / "selection_errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    filename = f"topic_{topic_index:03d}_task_{task_index:03d}.txt"
    payload = f"Task: {task_title}\n\n{content}"
    (errors_dir / filename).write_text(payload, encoding="utf-8")
def _watermark_image(path: Path, index: int) -> None:
    """Add a visible index watermark to the image."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        box_size = 28
        draw.rectangle((0, 0, box_size - 1, box_size - 1), fill=(0, 0, 0))
        font = ImageFont.load_default()
        label = str(index)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = max(0, (box_size - text_width) // 2)
        text_y = max(0, (box_size - text_height) // 2)
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
        image.save(path)


def _filter_similar_hashes(hashes: list[str]) -> list[str]:
    """Filter consecutive hashes with low Hamming distance."""
    filtered: list[str] = []
    previous: str | None = None
    for hash_value in hashes:
        if previous is None:
            filtered.append(hash_value)
            previous = hash_value
            continue
        if _hamming_distance(previous, hash_value) < 3:
            continue
        filtered.append(hash_value)
        previous = hash_value
    return filtered


def _hamming_distance(left: str, right: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 0


def _cleanup_temp_images(paths: list[Path]) -> None:
    """Remove temporary images after a task is processed."""
    for path in paths:
        path.unlink(missing_ok=True)


def _build_prompt(topic: CombinedTopic, task: CombinedTask, hashes: list[str]) -> str:
    """Build the prompt for a single task selection."""
    context_lines = []
    for segment in task.speech_segments:
        context_lines.append(f"- [{segment.start:.2f}-{segment.end:.2f}] {segment.text}")
    context_block = "\n".join(context_lines) if context_lines else "No transcript context."
    images_block = ", ".join(str(idx) for idx in range(len(hashes))) or "None"
    return PROMPT_TEMPLATE.format(
        topic=topic.topic,
        task=task.task,
        context_block=context_block,
        images_block=images_block,
    )


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


def _resolve_db_path(combine_json_path: Path, video_name: str) -> Path:
    """Resolve the path to the hash frame database."""
    video_stem = Path(video_name).stem if video_name else combine_json_path.stem
    return combine_json_path.parent / video_stem / "frames.sqlite3"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the selection command.

    Used by the CLI entry point for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Select keyframes for each task using a multimodal model.",
    )
    parser.add_argument("combine_json", type=Path, help="Combine JSON file")
    parser.add_argument("output_dir", type=Path, help="Output folder")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to the INI config file.",
    )
    return parser.parse_args()


def parse_config(config: configparser.ConfigParser) -> SelectionConfig:
    """Build a SelectionConfig from a loaded configparser instance."""
    defaults = SelectionConfig()
    section = config["selection"] if config.has_section("selection") else {}
    return SelectionConfig(
        model_name=str(section.get("model_name", defaults.model_name)),
        temperature=get_float(section, "temperature", defaults.temperature),
        context_window=get_int(section, "context_window", defaults.context_window),
        num_predict=get_int(section, "num_predict", defaults.num_predict),
        image_kind=str(section.get("image_kind", defaults.image_kind)),
        image_format=str(section.get("image_format", defaults.image_format)),
        max_retries=get_int(section, "max_retries", defaults.max_retries),
        pool_size=get_int(section, "pool_size", defaults.pool_size),
        request_delay=get_float(section, "request_delay", defaults.request_delay),
        retry_backoff=get_float(section, "retry_backoff", defaults.retry_backoff),
    )


def main(config_path: Path | None = None) -> None:
    """Run selection from the command line.

    This is the CLI entry point used by ``python -m modules.selection``.

    Args:
        config_path: Optional path to the INI config file.
    """
    args = _parse_args()
    parser = configparser.ConfigParser()
    path = args.config or config_path
    if path is not None and path.exists():
        parser.read(path, encoding="utf-8")
    config = parse_config(parser)
    select_keyframes(args.combine_json, args.output_dir, config)


if __name__ == "__main__":
    main()
