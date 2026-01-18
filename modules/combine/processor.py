from __future__ import annotations

import argparse
import configparser
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CombineConfig:
    """Configuration for combining module outputs."""


@dataclass(frozen=True)
class SpeechSegment:
    """Represents a transcribed speech segment in seconds."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SpeechDocument:
    """Speech module output document schema."""

    video: str
    segments: list[SpeechSegment]


@dataclass(frozen=True)
class TaskItem:
    """Represents a task item within a topic."""

    start: float
    task: str


@dataclass(frozen=True)
class TopicItem:
    """Represents a topic segment."""

    start: float
    topic: str
    tasks: list[TaskItem]


@dataclass(frozen=True)
class TopicsDocument:
    """Topics module output document schema."""

    video: str
    topics: list[TopicItem]


@dataclass(frozen=True)
class FrameHash:
    """Represents a hashed keyframe in seconds."""

    timestamp: float
    hash: str


@dataclass(frozen=True)
class HashDocument:
    """Hash module output document schema."""

    video: str
    frames: list[FrameHash]


@dataclass(frozen=True)
class CombinedTask:
    """Represents a combined task with aligned context."""

    start: float
    task: str
    speech_segments: list[SpeechSegment]
    keyframes: list[str]


@dataclass(frozen=True)
class CombinedTopic:
    """Represents a combined topic."""

    start: float
    topic: str
    tasks: list[CombinedTask]


@dataclass(frozen=True)
class CombinedDocument:
    """Combined output document schema."""

    video: str
    topics: list[CombinedTopic]


def combine_outputs(
    topics_json_path: Path,
    speech_json_path: Path,
    hash_json_path: Path,
    output_dir: Path,
    config: CombineConfig,
) -> CombinedDocument:
    """Combine topics, speech, and hash outputs into a single JSON file.

    Args:
        topics_json_path: Path to the topics module JSON output.
        speech_json_path: Path to the speech module JSON output.
        hash_json_path: Path to the hash module JSON output.
        output_dir: Output folder for the generated JSON file.
        config: Configuration for combining outputs.

    Output:
        Writes ``<stem>_combine.json`` into the output directory. The JSON
        document follows ``CombinedDocument``.

    Returns:
        The combined document with tasks enriched with speech and keyframes.

    Raises:
        FileNotFoundError: If any input JSON file does not exist.
        ValueError: If input JSON schema is invalid.
    """
    for path in (topics_json_path, speech_json_path, hash_json_path):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)

    topics_doc = _load_topics_document(topics_json_path)
    speech_doc = _load_speech_document(speech_json_path)
    hash_doc = _load_hash_document(hash_json_path)

    video_name = topics_doc.video
    video_stem = Path(video_name).stem if video_name else topics_json_path.stem
    json_path = output_dir / f"{video_stem}_combine.json"

    end_time = _compute_end_time(speech_doc.segments, hash_doc.frames)
    combined_topics: list[CombinedTopic] = []
    topics_sorted = sorted(topics_doc.topics, key=lambda topic: topic.start)
    for topic_index, topic in enumerate(topics_sorted):
        topic_end = _next_topic_start(topics_sorted, topic_index, end_time)
        combined_tasks = _combine_tasks(
            topic=topic,
            topic_end=topic_end,
            speech_segments=speech_doc.segments,
            frames=hash_doc.frames,
        )
        combined_topics.append(
            CombinedTopic(start=topic.start, topic=topic.topic, tasks=combined_tasks),
        )

    final_topics = _merge_empty_tasks_across_topics(combined_topics)
    output_doc = CombinedDocument(video=video_name, topics=final_topics)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(dataclasses.asdict(output_doc), handle, indent=2)

    return output_doc


def _load_topics_document(path: Path) -> TopicsDocument:
    """Load topics JSON document from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    video = data.get("video")
    topics = data.get("topics", [])
    if not isinstance(video, str) or not isinstance(topics, list):
        raise ValueError("Invalid topics JSON format")
    parsed_topics: list[TopicItem] = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        tasks = topic.get("tasks", [])
        parsed_tasks: list[TaskItem] = []
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                parsed_tasks.append(
                    TaskItem(
                        start=float(task.get("start", 0.0)),
                        task=str(task.get("task", "")),
                    ),
                )
        parsed_topics.append(
            TopicItem(
                start=float(topic.get("start", 0.0)),
                topic=str(topic.get("topic", "")),
                tasks=parsed_tasks,
            ),
        )
    return TopicsDocument(video=video, topics=parsed_topics)


def _load_speech_document(path: Path) -> SpeechDocument:
    """Load speech JSON document from disk."""
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


def _load_hash_document(path: Path) -> HashDocument:
    """Load hash JSON document from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    video = data.get("video")
    frames = data.get("frames", [])
    if not isinstance(video, str) or not isinstance(frames, list):
        raise ValueError("Invalid hash JSON format")
    parsed_frames: list[FrameHash] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        parsed_frames.append(
            FrameHash(
                timestamp=float(frame.get("timestamp", 0.0)),
                hash=str(frame.get("hash", "")),
            )
        )
    return HashDocument(video=video, frames=parsed_frames)


def _compute_end_time(segments: list[SpeechSegment], frames: list[FrameHash]) -> float:
    """Compute the end time for task windows."""
    last_segment_end = max((segment.end for segment in segments), default=0.0)
    last_frame_time = max((frame.timestamp for frame in frames), default=0.0)
    return max(last_segment_end, last_frame_time)


def _next_topic_start(
    topics: list[TopicItem],
    index: int,
    end_time: float,
) -> float:
    """Get the end time for the current topic based on the next topic."""
    if index + 1 < len(topics):
        return topics[index + 1].start
    return end_time


def _combine_tasks(
    topic: TopicItem,
    topic_end: float,
    speech_segments: list[SpeechSegment],
    frames: list[FrameHash],
) -> list[CombinedTask]:
    """Combine tasks for a topic with aligned speech and keyframes."""
    tasks_sorted = sorted(topic.tasks, key=lambda task: task.start)
    if not tasks_sorted:
        tasks_sorted = [TaskItem(start=topic.start, task="Overview of this section")]
    elif topic.start < tasks_sorted[0].start:
        tasks_sorted.insert(
            0,
            TaskItem(start=topic.start, task="Overview of this section"),
        )
    combined_tasks: list[CombinedTask] = []
    for task_index, task in enumerate(tasks_sorted):
        task_end = _next_task_start(tasks_sorted, task_index, topic_end)
        task_segments = _segments_in_window(speech_segments, task.start, task_end)
        task_hashes = _unique_hashes_in_window(frames, task.start, task_end)
        combined_tasks.append(
            CombinedTask(
                start=task.start,
                task=task.task,
                speech_segments=task_segments,
                keyframes=task_hashes,
            ),
        )
    return combined_tasks


def _merge_empty_tasks_across_topics(
    topics: list[CombinedTopic],
) -> list[CombinedTopic]:
    """Remove empty tasks, merging their keyframes into the previous task."""
    merged_topics: list[CombinedTopic] = []
    last_task_location: tuple[int, int] | None = None
    for topic in topics:
        merged_tasks: list[CombinedTask] = []
        for task in topic.tasks:
            if task.speech_segments:
                merged_tasks.append(task)
                last_task_location = (len(merged_topics), len(merged_tasks) - 1)
                continue
            if last_task_location is None:
                continue
            topic_idx, task_idx = last_task_location
            target_topic = merged_topics[topic_idx]
            target_task = target_topic.tasks[task_idx]
            merged_keyframes = _merge_keyframes(target_task.keyframes, task.keyframes)
            updated_task = CombinedTask(
                start=target_task.start,
                task=target_task.task,
                speech_segments=target_task.speech_segments,
                keyframes=merged_keyframes,
            )
            updated_tasks = list(target_topic.tasks)
            updated_tasks[task_idx] = updated_task
            merged_topics[topic_idx] = CombinedTopic(
                start=target_topic.start,
                topic=target_topic.topic,
                tasks=updated_tasks,
            )
        if merged_tasks:
            merged_topics.append(
                CombinedTopic(start=topic.start, topic=topic.topic, tasks=merged_tasks),
            )
    return merged_topics


def _merge_keyframes(existing: list[str], additional: list[str]) -> list[str]:
    """Merge keyframe hashes while preserving order and uniqueness."""
    merged: list[str] = []
    seen: set[str] = set()
    for value in existing + additional:
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged


def _next_task_start(tasks: list[TaskItem], index: int, end_time: float) -> float:
    """Get the end time for a task based on the next task or topic end."""
    if index + 1 < len(tasks):
        return tasks[index + 1].start
    return end_time


def _segments_in_window(
    segments: list[SpeechSegment],
    start: float,
    end: float,
) -> list[SpeechSegment]:
    """Select speech segments that start within the time window."""
    return [
        segment
        for segment in segments
        if segment.start >= start and segment.start < end
    ]


def _unique_hashes_in_window(
    frames: list[FrameHash],
    start: float,
    end: float,
) -> list[str]:
    """Select unique hashes for frames whose timestamps are within the window."""
    seen: set[str] = set()
    hashes: list[str] = []
    for frame in frames:
        if frame.timestamp < start or frame.timestamp >= end:
            continue
        if frame.hash in seen:
            continue
        seen.add(frame.hash)
        hashes.append(frame.hash)
    return hashes


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the combine command.

    Used by the CLI entry point for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Combine topics, speech, and hash outputs into one JSON file.",
    )
    parser.add_argument("topics_json", type=Path, help="Topics JSON file")
    parser.add_argument("speech_json", type=Path, help="Speech JSON file")
    parser.add_argument("hash_json", type=Path, help="Hash JSON file")
    parser.add_argument("output_dir", type=Path, help="Output folder")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to the INI config file.",
    )
    return parser.parse_args()


def parse_config(config: configparser.ConfigParser) -> CombineConfig:
    """Build a CombineConfig from a loaded configparser instance."""
    _ = config
    return CombineConfig()


def main(config_path: Path | None = None) -> None:
    """Run combine from the command line.

    This is the CLI entry point used by ``python -m modules.combine``.

    Args:
        config_path: Optional path to the INI config file.
    """
    args = _parse_args()
    parser = configparser.ConfigParser()
    path = args.config or config_path
    if path is not None and path.exists():
        parser.read(path, encoding="utf-8")
    config = parse_config(parser)
    combine_outputs(
        args.topics_json,
        args.speech_json,
        args.hash_json,
        args.output_dir,
        config,
    )


if __name__ == "__main__":
    main()
