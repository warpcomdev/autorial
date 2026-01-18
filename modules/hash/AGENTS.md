This module processes a video file and generates a sequence of keyframes with a hash obtained using the python imagehash library.

ffmpeg is used to extract two streams of key frames from the video: one frame per second at the original resolution, and another frame per second resizing the video to a configurable size. The default resized size must be 1008 x 758 resolution. This is close to the 1024 x 768 resolution, but makes frames divisible by 28, which is the patch size for qwen3-vl.

The script generates a JSON file with the name of the video, and a list of objects with the timestamp of the frame plus references to the hash database entries for the original and resized frames.

Policy decisions:
- Keyframes are stored in a SQLite database (`frames.sqlite3`) under the per-video output folder; this database is rebuilt on each run.
- Extracted frames are written to temporary folders and removed after upload to the database.
- Original frames are extracted with `setsar=1` to normalize pixel aspect ratio and avoid stretched images.
