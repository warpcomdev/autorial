from __future__ import annotations

import argparse
import configparser
import logging
from pathlib import Path

from modules import combine as combine_module
from modules import hash as hash_module
from modules import markdown as markdown_module
from modules import selection as selection_module
from modules import speech as speech_module
from modules import topics as topics_module


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Run the autorial pipeline on a video file.",
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
        "--continue-from",
        dest="continue_from",
        choices=["speech", "hash", "topics", "combine", "selection", "markdown"],
        help="Skip stages before this one and continue from the specified stage.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full processing pipeline."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = configparser.ConfigParser()
    config_path = args.config or Path(__file__).parent / "config.ini"
    if config_path.exists():
        parser.read(config_path, encoding="utf-8")

    stages = ["speech", "hash", "topics", "combine", "selection", "markdown"]
    start_index = 0
    if args.continue_from:
        start_index = stages.index(args.continue_from)

    video_path = args.video
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    video_stem = video_path.stem
    speech_json = output_dir / f"{video_stem}_speech.json"
    hash_json = output_dir / f"{video_stem}_hash.json"
    topics_json = output_dir / f"{video_stem}_speech_topics.json"
    combine_json = output_dir / f"{video_stem}_combine.json"
    selection_json = output_dir / f"{video_stem}_combine_selection.json"

    if start_index <= stages.index("speech"):
        logging.info("Running speech module...")
        speech_config = speech_module.parse_config(parser)
        speech_module.transcribe_video(video_path, output_dir, speech_config)

    if start_index <= stages.index("hash"):
        logging.info("Running hash module...")
        hash_config = hash_module.parse_config(parser)
        hash_module.hash_video_frames(video_path, output_dir, hash_config)

    if start_index <= stages.index("topics"):
        logging.info("Running topics module...")
        topics_config = topics_module.parse_config(parser)
        topics_module.generate_topics(speech_json, output_dir, topics_config)

    if start_index <= stages.index("combine"):
        logging.info("Running combine module...")
        combine_config = combine_module.parse_config(parser)
        combine_module.combine_outputs(
            topics_json,
            speech_json,
            hash_json,
            output_dir,
            combine_config,
        )

    if start_index <= stages.index("selection"):
        logging.info("Running selection module...")
        selection_config = selection_module.parse_config(parser)
        selection_module.select_keyframes(combine_json, output_dir, selection_config)

    if start_index <= stages.index("markdown"):
        logging.info("Running markdown module...")
        markdown_config = markdown_module.parse_config(parser)
        markdown_module.generate_markdown(
            combine_json,
            selection_json,
            output_dir,
            markdown_config,
        )


if __name__ == "__main__":
    main()
