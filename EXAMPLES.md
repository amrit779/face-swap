# Examples

These examples use placeholder file names. Use your own media only when you have permission to process and publish it.

## Example Input Layout

```text
face_swap/
|-- models/
|   |-- GFPGANv1.3.pth
|   `-- inswapper_128.onnx
|-- source_face.jpg
|-- target/
|   |-- interview_clip.mp4
|   `-- rehearsal_take.mov
|-- output/
|-- requirements.txt
`-- swap_script.py
```

## Preflight Example

Command:

```bash
FACE_SWAP_LOG_LEVEL=INFO ./swap_env/bin/python swap_script.py --preflight
```

Expected result:

- validates the source image, target path, output path, and model files
- checks source face detection
- checks target video metadata and writer/export support
- exits before processing a full video

## Single Video Example

Command:

```bash
FACE_SWAP_LOG_LEVEL=INFO ./swap_env/bin/python swap_script.py \
  --source source_face.jpg \
  --target target/interview_clip.mp4 \
  --output output/interview_clip_swapped.mp4 \
  --det-size 640 640 \
  --detection-interval 2 \
  --enhancement-interval 2 \
  --export-crf 18
```

Expected output:

```text
output/
`-- interview_clip_swapped.mp4
```

The output video keeps the target video's audio track when ffmpeg can read and copy it. Temporary files with `.incomplete` or processing markers may appear during the run and should be removed automatically after successful export.

## Batch Directory Example

Command:

```bash
FACE_SWAP_LOG_LEVEL=INFO ./swap_env/bin/python swap_script.py \
  --source source_face.jpg \
  --target target \
  --output output \
  --workers 2 \
  --skip-enhancement \
  --progress-every 10
```

Expected output:

```text
output/
|-- interview_clip.mp4
`-- rehearsal_take.mov
```

In directory mode, each successfully processed input video is deleted from `target/` after the final output is written. Keep a backup of originals before running batch jobs.

## Quality Example

Command:

```bash
FACE_SWAP_LOG_LEVEL=INFO ./swap_env/bin/python swap_script.py \
  --source source_face.jpg \
  --target target/interview_clip.mp4 \
  --output output/interview_clip_hq.mp4 \
  --det-size 640 640 \
  --detection-interval 1 \
  --enhancement-interval 1 \
  --identity-threshold 0.35 \
  --color-match-strength 0.45 \
  --track-smoothing 0.65 \
  --export-scale 1.25 \
  --export-crf 16 \
  --export-preset slow
```

Expected output:

```text
output/
`-- interview_clip_hq.mp4
```

This path is slower because it detects more often, enhances every swapped frame, and runs a final super-resolution export pass.
