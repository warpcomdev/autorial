This module combines the results of the "topics", "speech", and "hash" modules into a single JSON file for easier consumption by downstream applications.

It reads the output JSON files from the three modules and merges them based on timestamps. The combined output includes topics with their associated tasks, speech segments, and keyframe hashes.

The output is a JSON file containing an array of topic objects, each with a "start" field (timestamp in seconds), a "topic" field (text description), and a "tasks" field. The "tasks" field in turn contains "start", "task", "speech_segments" and "keyframes".

The key frames and speech segments are matched to tasks based on their timestamps. Each task will include the speech segments and key frames that fall within its time range (from the task start, to the next task, or the end of the audio).

Policy decisions:
- If a topic has no tasks, or there is a gap between the topic start and the first task, a synthetic task "Overview of this section" is inserted at the topic start.
- Tasks with no speech segments are removed by merging their keyframes into the previous task in the same topic when possible; if the empty task is the first task in the topic, its keyframes merge into the last task of the previous topic.
