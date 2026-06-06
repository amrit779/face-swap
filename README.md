# High-Fidelity Face Swap for macOS

This project swaps a face from a source image into one or more target videos using InsightFace for detection and identity embeddings, then optionally restores selected swapped frames with GFPGAN.

Use this project only with consent and clear disclosure. See [Responsible Use](RESPONSIBLE_USE.md) before processing or sharing generated media.

The current pipeline is tuned for Apple Silicon and includes:

- CoreML-first ONNX Runtime execution when available
- target identity locking using face embeddings
- optical-flow tracking between full detections
- landmark-based face masking for local color correction and blending
- post-process GFPGAN enhancement on selected swapped frames
- optional final RealESRGAN super-resolution pass
- final ffmpeg export with original audio preserved

## Requirements

- macOS on Apple Silicon
- Python 3.10
- Homebrew
- ffmpeg and ffprobe available on PATH
- source image file
- one target video or a target directory containing videos
- model files in the models directory

## Setup

Install Python 3.10 and create the virtual environment:

```bash
brew install python@3.10 ffmpeg
python3.10 -m venv swap_env
source swap_env/bin/activate
```

Install dependencies:

```bash
pip install "setuptools<82"
pip install -r requirements.txt
```

## Models

Place these files in the models directory:

- models/inswapper_128.onnx
	Download: https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx
- models/GFPGANv1.3.pth
	Download: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth

Example:

```bash
mkdir -p models
curl -L https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx -o models/inswapper_128.onnx
curl -L https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -o models/GFPGANv1.3.pth
```

RealESRGAN weights download automatically on first use unless you provide `--superres-model-path`.

Expected layout:

```text
.
├── models/
│   ├── GFPGANv1.3.pth
│   └── inswapper_128.onnx
├── source_face.jpg
├── target/
│   ├── clip1.mp4
│   └── clip2.mp4
├── output/
├── requirements.txt
└── swap_script.py
```

## Basic Run

Default run:

```bash
./swap_env/bin/python swap_script.py
```

By default, the script expects:

- `source_face.jpg` as the source image
- `target/` as the input directory
- `output/` as the output directory

Each output file keeps the same basename as its source video.
In directory mode, each source video is deleted from `target/` after its output is finalized successfully.
Use `--workers N` to process multiple target videos in parallel when running against a directory.

Single-video example:

```bash
./swap_env/bin/python swap_script.py \
	--source my_face.jpg \
	--target clip.mp4 \
	--output swapped.mp4
```

Batch directory example:

```bash
./swap_env/bin/python swap_script.py \
	--source source_face.jpg \
	--target target \
	--output output
```

More example input and output layouts are available in [EXAMPLES.md](EXAMPLES.md).

## Preflight

Use preflight mode to validate models, inputs, source face detection, video metadata, writer support, and RealESRGAN setup without processing full videos:

```bash
./swap_env/bin/python swap_script.py --preflight
```

## Pipeline

For each target video, the script runs these stages:

1. Swap pass: detects or tracks the target face, locks onto one identity using embeddings, swaps the face, and applies landmark-based local color correction.
2. Enhancement pass: optionally reopens the swapped video and runs GFPGAN on selected swapped frames only.
3. Super-resolution pass: optionally reopens the processed video and runs RealESRGAN on every frame.
4. Final export pass: exports the processed video with ffmpeg and preserves the original audio track.

## Quality Controls

Relevant options:

- `--det-size WIDTH HEIGHT`: larger detection size can help difficult faces but costs speed.
- `--detection-interval N`: `1` means detect every frame. Higher values rely more on tracking.
- `--enhancement-interval N`: `1` enhances every swapped frame. Higher values enhance fewer frames and can introduce visible flicker.
- `--enhancement-workers N`: number of CPU worker processes for the GFPGAN enhancement pass on a single video. This preserves frame order and is automatically disabled when batch video workers are already in use.
- `--identity-threshold X`: minimum embedding similarity required to keep swapping the same person.
- `--color-match-strength X`: strength of landmark-masked local face color correction from `0.0` to `1.0`.
- `--track-smoothing X`: temporal smoothing for tracked face geometry from `0.0` to `0.95`. Higher values reduce jitter.
- `--superres-model NAME`: RealESRGAN model used for final video super-resolution.
- `--superres-model-path PATH_OR_URL`: optional custom RealESRGAN weights path or URL.
- `--workers N`: number of parallel worker processes for directory-based batch runs.
- `--export-scale X`: RealESRGAN outscale factor. `1.0` skips the super-resolution pass.
- `--export-crf N`: lower CRF means higher visual quality in the final x264 export.
- `--export-preset NAME`: x264 preset for the final export pass.
- `--skip-enhancement`: disables GFPGAN to prioritize speed.

## Recommended Commands

Fast preview batch run:

```bash
FACE_SWAP_LOG_LEVEL=INFO ./swap_env/bin/python swap_script.py \
	--skip-enhancement \
	--target target \
	--output output \
	--workers 2 \
	--det-size 320 320 \
	--detection-interval 4 \
	--progress-every 10
```

Balanced run:

```bash
FACE_SWAP_LOG_LEVEL=INFO ./swap_env/bin/python swap_script.py \
	--target target \
	--output output \
	--det-size 640 640 \
	--detection-interval 2 \
	--enhancement-interval 2 \
	--enhancement-workers 2 \
	--identity-threshold 0.35 \
	--color-match-strength 0.35 \
	--export-crf 18 \
	--export-preset slow
```

Highest-quality run in the current pipeline:

```bash
FACE_SWAP_LOG_LEVEL=INFO ./swap_env/bin/python swap_script.py \
	--target target \
	--output output \
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

## Logging

The script logs:

- input validation
- model loading
- source face detection
- target video metadata
- batch progress across discovered videos
- swap-pass progress with elapsed time, speed, and ETA
- enhancement-pass progress with enhanced-frame counts
- super-resolution pass progress
- final export behavior, including audio preservation

Debug mode:

```bash
FACE_SWAP_LOG_LEVEL=DEBUG ./swap_env/bin/python swap_script.py
```

## Notes

- The first run may download InsightFace support models into `~/.insightface`.
- GFPGAN dependencies may also download face parsing and detection weights on first use.
- RealESRGAN weights may also download on first use if you do not provide a local weights file.
- The script includes a compatibility shim for newer torchvision releases so GFPGAN and RealESRGAN import cleanly.
- The final output keeps the original audio track by exporting with ffmpeg.
- Final export quality depends on the input video quality. Re-encoding cannot restore detail that is not present in the source.

## Project Documents

- [License](LICENSE)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Responsible Use](RESPONSIBLE_USE.md)
- [GitHub Topics](GITHUB_TOPICS.md)

## Troubleshooting

If the script fails immediately:

- Confirm the source image exists.
- Confirm the target file or target directory exists.
- Confirm the GFPGAN and inswapper model files exist in the models directory.
- Confirm you are using the Python 3.10 virtual environment.
- Confirm `ffmpeg` and `ffprobe` are installed and on PATH.

If the wrong person is swapped:

- Lower `--detection-interval` so full detection runs more often.
- Increase `--identity-threshold` slightly, for example from `0.35` to `0.40`.
- Use a target clip with fewer simultaneous faces.

If the swap looks unstable:

- Set `--detection-interval 1`.
- Keep `--enhancement-interval 1` so GFPGAN does not change appearance only on some frames.
- Increase `--track-smoothing` slightly, for example from `0.65` to `0.75`, if the face jitters.
- Use a clearer source photo with a more frontal face.
- Use a less compressed target video.

If output quality is weak:

- Use `--enhancement-interval 1`.
- Increase `--color-match-strength` carefully.
- Lower `--export-crf` to `16` or `14`.
- Try `--export-scale 1.25` or `1.5` for a cleaner final presentation.

If the swap looks unstable:

- Set `--detection-interval 1`.
- Use a clearer source photo with a more frontal face.
- Use a less compressed target video.

If output quality is weak:

- Use `--enhancement-interval 1`.
- Increase `--color-match-strength` carefully.
- Lower `--export-crf` to `16` or `14`.
- Try `--export-scale 1.25` or `1.5` for a cleaner final presentation.
# High-Fidelity Face Swap for macOS

This project swaps a face from a source image into a target video using InsightFace for detection and swapping, then runs GFPGAN to restore the swapped face.

The current script is tuned for local macOS execution with Python 3.10 and Apple Silicon. It prefers CoreML for InsightFace ONNX inference when available, loads only the InsightFace modules needed for swapping, tracks faces between detection passes, and uses MPS for GFPGAN when available.
## Run

Default run:
## Requirements
- macOS on Apple Silicon
- Python 3.10
- Homebrew
- A source image named source_face.jpg
Custom input and output paths:
- Model files in the models directory

./swap_env/bin/python swap_script.py \
	--source my_face.jpg \
	--target clip.mp4 \
	--output swapped.mp4


```bash
./swap_env/bin/python swap_script.py --preflight
```

This is useful before long runs or when testing a new input video.

## Common Options

	--progress-every 25
```

Useful flags:

- `--provider NAME`: overrides ONNX Runtime provider order. Repeat the flag to set fallbacks.
- `--preflight`: runs validation only.


1. Swap pass: reads the source face once, runs face detection every few frames, and tracks the target face in between.
2. Enhancement pass: reopens the swapped video and applies GFPGAN only on selected swapped frames.
source swap_env/bin/activate
```

pip install -r requirements.txt
```

## Models

Place these files in the models directory:

- models/inswapper_128.onnx
- models/GFPGANv1.3.pth

Expected layout:

```text
.
├── models/
│   ├── GFPGANv1.3.pth
│   └── inswapper_128.onnx
├── source_face.jpg
├── target_video.mp4
├── requirements.txt
└── swap_script.py
```

## Run

With the environment activated:

```bash
python swap_script.py
```

Without activating the environment:

```bash
./swap_env/bin/python swap_script.py
```

The output video is written to final_output.mp4.

During processing, frames are written to a temporary file ending in .incomplete.mp4. The final_output.mp4 file is only replaced after the run finishes successfully, so a half-written MP4 is no longer exposed as the final result.

## Logging

The script now emits structured logs for:

- input validation
- model loading
- source face detection
- video metadata
- frame processing progress with elapsed time, processing FPS, and ETA
- separate enhancement pass progress with enhanced-frame counts
- final output path

Use debug logging when you want per-frame diagnostics:

```bash
FACE_SWAP_LOG_LEVEL=DEBUG ./swap_env/bin/python swap_script.py
```

Example with faster settings:

```bash
FACE_SWAP_LOG_LEVEL=INFO ./swap_env/bin/python swap_script.py \
	--detection-interval 4 \
	--enhancement-interval 8 \
	--det-size 320 320 \
	--progress-every 10
```

Fastest swap-only example:

```bash
FACE_SWAP_LOG_LEVEL=INFO ./swap_env/bin/python swap_script.py \
	--skip-enhancement \
	--detection-interval 4 \
	--det-size 320 320 \
	--progress-every 10
```

## Notes

- The first run may download InsightFace support models into ~/.insightface.
- GFPGAN dependencies may also download face parsing and detection weights on first use.
- The script includes a compatibility shim for newer torchvision releases so GFPGAN and BasicSR can import cleanly.
- CoreML startup can still take time on the first load because ONNX Runtime may prepare or compile models for the provider.
- Post-process enhancement means the swap pass finishes sooner, but total runtime still depends on how many frames you choose to enhance.
- The final output keeps the original audio track by remuxing it with ffmpeg/ffprobe after video processing completes.

## Troubleshooting

If the script fails immediately:

- Confirm source_face.jpg and target_video.mp4 exist in the project root.
- Confirm both model files exist in the models directory.
- Confirm you are using Python 3.10 from the virtual environment.

If no face is swapped:

- Use a clear, front-facing source photo.
- Try a target video with larger and better-lit faces.

If processing is slow:

- The first run is slower because models and caches are initialized.
- CoreML-backed InsightFace startup still has a noticeable warm-up cost, but it avoids the much slower CPU-only path used before.
- Use `--skip-enhancement` if you need speed more than restoration quality.
- Try `--det-size 320 320` or `--det-size 416 416` to reduce detection cost.
- Increase `--detection-interval` to reduce how often full detection runs. Lower it again if tracking drifts.
- Increase `--enhancement-interval` to enhance fewer swapped frames in the second pass.
- If you want to compare providers, run with `--provider CPUExecutionProvider` to force CPU-only ONNX inference.
