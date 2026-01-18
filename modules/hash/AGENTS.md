This module processes a video file and generates a sequence of keyframes with a hash obtained using the python imagehash library.

ffmpeg is used to extract two streams of key frames from the video: onme frame per second at the original resolution, and another frame per second resizing the video to a configurable size. The default resized size must be 1008 x 758 resolution. This is close to the 1024 x 768 resolution, but makes frames divisible by 28, which is the patch size for qwen3-vl. 

The script generates json file with the name of the video, and a list of objects with the timestamp of the frame, the name of the file holding the original resolution keyframe, the name of the file holding the resized keyframe, and a hash calculated with the imagehash python library.

All output images are stored in a subdirectory of the output folder, named after the input video.
