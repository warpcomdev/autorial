# Autorial

Autorial is a modular pipeline that turns screen‑recorded training videos into structured documentation. It extracts speech, topics, keyframes, tasks, and produces a tutorial‑style markdown guide with relevant screenshots.

The pipeline is designed to be transparent: each module writes intermediate JSON and the keyframes database, so you can inspect, debug, and replace steps as needed.

## Quick start

Prerequisites:
- Python 3.11
- `ffmpeg` and `ffprobe` on your PATH
- Ollama running locally for the LLM‑powered modules

Install dependencies:
```
uv sync
```

Run the full pipeline:
```
uv run python main.py input/sample_video.mp4 output -c config.ini
```

Resume from a stage:
```
uv run python main.py input/sample_video.mp4 output -c config.ini --continue-from selection
```

## Output layout

Typical outputs in `output/`:
- `<video>_speech.json` – speech segments
- `<video>_hash.json` – keyframe hashes + DB references
- `<video>_speech_topics.json` – topic/task segmentation
- `<video>_combine.json` – merged topics/tasks with speech + keyframes
- `<video>_combine_selection.json` – selected screenshots per task
- `<video>/frames.sqlite3` – keyframe blobs
- `<video>/img/*.jpg` – extracted images used in markdown
- `<video>/section_0001.md` … – per‑section markdown
- `<video>/README.md` – section index with summaries

## Configuration

All configuration lives in `config.ini`. Paths are passed on the command line.

### `[speech]`
Transcription with Faster‑Whisper.

- `model_name`: Whisper model (e.g., `large-v3`)
- `batch_size`: GPU batch size
- `beam_size`: decoding beam size
- `compute_type`: `float16`, `int8`, `int8_float16`
- `language`: optional language code

### `[hash]`
Keyframe extraction and perceptual hashes.

- `fps`: sampling rate (frames per second)
- `image_format`: `jpg` recommended
- `hash_algorithm`: `phash`, `dhash`, `ahash`, `whash`
- `resized_width`, `resized_height`: resized keyframe dimensions

### `[topics]`
Topic/task segmentation from speech.

- `model_name`: Ollama model (default `gpt-oss:20b`)
- `temperature`: usually `0.0`
- `context_window`: context size
- `num_predict`: max output tokens

### `[selection]`
Multimodal selection of relevant screenshots per task.

- `model_name`: vision model (default `qwen3-vl:4b`)
- `temperature`: usually `0.0`
- `context_window`, `num_predict`: token limits
- `image_kind`: `resized` or `original`
- `image_format`: `jpg`
- `max_retries`: retry on empty responses
- `pool_size`: worker processes
- `request_delay`, `retry_backoff`: throttling

### `[markdown]`
Markdown generation and summaries.

- `model_name`: LLM for writing (default `gpt-oss:20b`)
- `temperature`: usually `0.0`
- `context_window`, `num_predict`: token limits for section output
- `summary_num_predict`: token limit for section summaries
- `image_kind`: `original` recommended
- `image_format`: `jpg`

## How to record for best results

The quality of outputs depends heavily on how the video is narrated:

- Introduce each **section** clearly with a short statement of intent.
- State each **task** as an explicit action (e.g., “Now create a new realm…”).
- When a task has multiple actions, verbalize each step in order.
- Pause briefly between tasks to help the models detect boundaries.
- Name key UI elements aloud (buttons, tabs, menu items).
- Avoid long tangents that span multiple topics; it blurs segmentation.
- Keep the cursor steady when possible so screenshots are visually stable.

## Running modules individually

Each module has a CLI entry point:

```
python -m modules.speech <video> <output_dir> -c config.ini
python -m modules.hash <video> <output_dir> -c config.ini
python -m modules.topics <speech_json> <output_dir> -c config.ini
python -m modules.combine <topics_json> <speech_json> <hash_json> <output_dir> -c config.ini
python -m modules.selection <combine_json> <output_dir> -c config.ini
python -m modules.markdown <combine_json> <selection_json> <output_dir> -c config.ini
```

## License

TBD
