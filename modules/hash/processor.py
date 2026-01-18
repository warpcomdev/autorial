from __future__ import annotations

import argparse
import configparser
import dataclasses
import json
import logging
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import imagehash

from modules.config import get_float, get_int


@dataclass(frozen=True)
class HashConfig:
    """Configuration for keyframe hashing.

    Attributes:
        fps: Frames per second to sample from the input video.
        image_format: Output image format for extracted frames.
        hash_algorithm: Name of the imagehash algorithm to use.
        resized_width: Width for resized keyframes.
        resized_height: Height for resized keyframes.
    """

    fps: float = 1.0
    image_format: str = "jpg"
    hash_algorithm: str = "phash"
    resized_width: int = 1008
    resized_height: int = 758


@dataclass(frozen=True)
class FrameHash:
    """Represents a hashed keyframe in seconds."""

    timestamp: float
    original_frame: "FrameReference"
    resized_frame: "FrameReference"
    hash: str


@dataclass(frozen=True)
class FrameReference:
    """Reference to a frame blob stored in SQLite."""

    db_path: str
    kind: str
    hash: str


@dataclass(frozen=True)
class OutputDocument:
    """Output document schema for keyframe hashing results.

    Attributes:
        video: Input video filename.
        frames: Keyframe hash metadata.
    """

    video: str
    frames: list[FrameHash]


def hash_video_frames(
    video_path: Path,
    output_dir: Path,
    config: HashConfig,
) -> OutputDocument:
    """Extract keyframes, hash them, and write a JSON output.

    Args:
        video_path: Path to the input video file.
        output_dir: Output folder for the generated JSON file.
        config: Configuration for keyframe hashing.

    Output:
        Writes ``<stem>.json`` into the output directory. The JSON document
        follows ``OutputDocument``. Keyframes are extracted to temporary
        folders, stored in a SQLite database, and removed from disk. When
        multiple frames share the same hash, the latest frame overwrites
        earlier blobs.

    Returns:
        The output document describing hashed keyframes. All frame references
        are kept in the JSON, but only the latest frame per hash remains in
        the SQLite store.

    Raises:
        FileNotFoundError: If the input video file does not exist.
        RuntimeError: If ffmpeg is not available.
        ValueError: If an unsupported hash algorithm is requested.
    """
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to extract keyframes")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / video_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{video_path.stem}_hash.json"
    db_path = frames_dir / "frames.sqlite3"
    logging.info("Writing frame blobs to %s", db_path)
    if db_path.exists():
        db_path.unlink()
    connection = _init_db(db_path)

    with tempfile.TemporaryDirectory() as temp_root:
        temp_path = Path(temp_root)
        original_dir = temp_path / "original"
        resized_dir = temp_path / "resized"
        original_dir.mkdir(parents=True, exist_ok=True)
        resized_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Extracting original keyframes...")
        frame_entries = _extract_keyframes(video_path, original_dir, config)
        logging.info("Extracting resized keyframes...")
        resized_map = _extract_resized_frames(video_path, resized_dir, config)
        hasher = _select_hasher(config.hash_algorithm)
        frame_hashes: list[FrameHash] = []
        total_frames = len(frame_entries)
        for idx, entry in enumerate(frame_entries, start=1):
            resized_path = resized_map.get(entry.index)
            if resized_path is None:
                msg = f"Missing resized frame for index {entry.index}"
                raise RuntimeError(msg)
            with Image.open(entry.path) as image:
                hash_value = str(hasher(image))
            _store_frame_blob(connection, hash_value, "original", entry.path)
            _store_frame_blob(connection, hash_value, "resized", resized_path)
            frame_hashes.append(
                FrameHash(
                    timestamp=entry.timestamp,
                    original_frame=FrameReference(
                        db_path=f"{video_path.stem}/frames.sqlite3",
                        kind="original",
                        hash=hash_value,
                    ),
                    resized_frame=FrameReference(
                        db_path=f"{video_path.stem}/frames.sqlite3",
                        kind="resized",
                        hash=hash_value,
                    ),
                    hash=hash_value,
                ),
            )
            if idx % 50 == 0 or idx == total_frames:
                logging.info("Hashed %d/%d frames", idx, total_frames)

    output_doc = OutputDocument(video=video_path.name, frames=frame_hashes)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(dataclasses.asdict(output_doc), handle, indent=2, ensure_ascii=False)

    connection.close()
    return output_doc


def _extract_resized_frames(
    video_path: Path,
    resized_dir: Path,
    config: HashConfig,
) -> dict[int, Path]:
    """Extract resized keyframes and return an index-to-path map."""
    image_format = config.image_format.lower().strip(".")
    output_pattern = str(resized_dir / f"frame_%06d.{image_format}")
    scale_filter = f"scale={config.resized_width}:{config.resized_height}"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={config.fps},{scale_filter}",
        "-vsync",
        "vfr",
        "-start_number",
        "0",
        "-loglevel",
        "info",
        output_pattern,
    ]
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return _map_frame_indices(resized_dir, image_format)


def _map_frame_indices(frames_dir: Path, image_format: str) -> dict[int, Path]:
    """Map frame index values to file paths in a frame directory."""
    mapping: dict[int, Path] = {}
    for frame in frames_dir.glob(f"frame_*.{image_format}"):
        match = re.search(r"frame_(\d+)", frame.stem)
        if not match:
            continue
        mapping[int(match.group(1))] = frame
    return mapping


def _init_db(db_path: Path) -> sqlite3.Connection:
    """Create or open the SQLite database for frame blobs."""
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS frames (
            hash TEXT NOT NULL,
            kind TEXT NOT NULL,
            data BLOB NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (hash, kind)
        )
        """
    )
    connection.commit()
    return connection


def _store_frame_blob(
    connection: sqlite3.Connection,
    hash_value: str,
    kind: str,
    path: Path,
) -> None:
    """Store a frame blob and remove the source file."""
    data = path.read_bytes()
    connection.execute(
        """
        INSERT INTO frames (hash, kind, data, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(hash, kind) DO UPDATE SET
            data=excluded.data,
            updated_at=excluded.updated_at
        """,
        (hash_value, kind, data, time.time()),
    )
    connection.commit()
    path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _FrameEntry:
    """Internal helper for extracted keyframes."""

    index: int
    timestamp: float
    path: Path


def _extract_keyframes(
    video_path: Path,
    frames_dir: Path,
    config: HashConfig,
) -> list[_FrameEntry]:
    """Extract keyframes using ffmpeg with showinfo timestamps."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to extract keyframes")

    image_format = config.image_format.lower().strip(".")
    output_pattern = str(frames_dir / f"frame_%06d.{image_format}")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={config.fps},showinfo,setsar=1",
        "-vsync",
        "vfr",
        "-start_number",
        "0",
        "-loglevel",
        "info",
        output_pattern,
    ]
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timestamps = _parse_showinfo_timestamps(result.stderr)
    if not timestamps:
        timestamps = _fallback_timestamps(frames_dir, config)
    timestamps = sorted(timestamps, key=lambda item: item[0])
    entries: list[_FrameEntry] = []
    for index, timestamp in timestamps:
        frame_path = frames_dir / f"frame_{index:06d}.{image_format}"
        if frame_path.exists():
            entries.append(_FrameEntry(index=index, timestamp=timestamp, path=frame_path))
    return entries


def _parse_showinfo_timestamps(output: str) -> list[tuple[int, float]]:
    """Parse showinfo output into (frame_index, timestamp) pairs."""
    pattern = re.compile(r"n:\s*(\d+).*pts_time:([0-9.]+)")
    timestamps: list[tuple[int, float]] = []
    for line in output.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        index = int(match.group(1))
        time = float(match.group(2))
        timestamps.append((index, time))
    return timestamps


def _fallback_timestamps(
    frames_dir: Path,
    config: HashConfig,
) -> list[tuple[int, float]]:
    """Fallback timestamps based on output ordering and fps."""
    fps = config.fps or 1.0
    image_format = config.image_format.lower().strip(".")
    frames = sorted(frames_dir.glob(f"frame_*.{image_format}"))
    entries: list[tuple[int, float]] = []
    for frame in frames:
        match = re.search(r"frame_(\d+)", frame.stem)
        if not match:
            continue
        index = int(match.group(1))
        entries.append((index, index / fps))
    return entries


def _select_hasher(algorithm: str):
    """Select an imagehash function by name."""
    key = algorithm.lower()
    options = {
        "phash": imagehash.phash,
        "dhash": imagehash.dhash,
        "ahash": imagehash.average_hash,
        "average_hash": imagehash.average_hash,
        "whash": imagehash.whash,
    }
    if key not in options:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    return options[key]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the hash command.

    Used by the CLI entry point for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Extract and hash keyframes from a video file.",
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("output_dir", type=Path, help="Output folder")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to the INI config file.",
    )
    return parser.parse_args()


def parse_config(config: configparser.ConfigParser) -> HashConfig:
    """Build a HashConfig from a loaded configparser instance.

    Input/output paths are provided at runtime; module tuning lives in config.ini.
    """
    defaults = HashConfig()
    section = config["hash"] if config.has_section("hash") else {}
    return HashConfig(
        fps=get_float(section, "fps", defaults.fps),
        image_format=str(section.get("image_format", defaults.image_format)),
        hash_algorithm=str(section.get("hash_algorithm", defaults.hash_algorithm)),
        resized_width=get_int(section, "resized_width", defaults.resized_width),
        resized_height=get_int(section, "resized_height", defaults.resized_height),
    )


def main(config_path: Path | None = None) -> None:
    """Run keyframe hashing from the command line.

    This is the CLI entry point used by ``python -m modules.hash``.

    Args:
        config_path: Optional path to the INI config file.
    """
    args = _parse_args()
    parser = configparser.ConfigParser()
    path = args.config or config_path
    if path is not None and path.exists():
        parser.read(path, encoding="utf-8")
    config = parse_config(parser)
    hash_video_frames(args.video, args.output_dir, config)


if __name__ == "__main__":
    main()
