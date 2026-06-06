# Contributing

Thanks for helping improve this face-swap pipeline. Contributions are welcome when they make the tool more reliable, transparent, respectful of consent, or easier to run safely.

## Ground Rules

- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Follow the [Responsible Use](RESPONSIBLE_USE.md) guidelines.
- Do not submit examples, tests, fixtures, or screenshots that contain a real person's face unless you have permission to use that media in this project.
- Prefer synthetic, public-domain, licensed, or self-owned media for examples.
- Keep model weights, generated videos, and large media files out of git.

## Development Setup

```bash
brew install python@3.10 ffmpeg
python3.10 -m venv swap_env
source swap_env/bin/activate
pip install "setuptools<82"
pip install -r requirements.txt
```

Place required model files under `models/` as described in [README.md](README.md).

## Before Opening a Pull Request

1. Run a preflight check:

   ```bash
   ./swap_env/bin/python swap_script.py --preflight
   ```

2. Test the command path you changed with a short local video.
3. Update [EXAMPLES.md](EXAMPLES.md) or [README.md](README.md) when changing user-facing behavior.
4. Make sure generated files, local models, virtual environments, and output videos are not committed.

## Pull Request Guidelines

- Explain the problem and the behavior change.
- Include the command you used to verify the change.
- Note any macOS, Apple Silicon, ffmpeg, ONNX Runtime, or model-specific assumptions.
- Keep unrelated refactors out of feature or bug-fix pull requests.
