This module processes a video file and generates and transcribes audio to text, using whisperxl.

The module is designed to run efficiently on a machine with a single nvidia RTX 3090 GPU, and use batching to maximize the usage of GPU memory.

The output is a JSON file containing an array of objects, each containing the transcription of a segment together with the start and end timestamps.
