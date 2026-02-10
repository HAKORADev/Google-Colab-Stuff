# Qwen3-TTS Voice Design Tool

Simple text-to-speech using Qwen3-TTS 1.7B VoiceDesign model.

## Usage

```bash
python main.py "Your text here" --voice "voice description"
```

## Examples

```bash
python main.py "Hello world" --voice "a clear adult male with professional tone"

python main.py "Welcome to the future of AI" --voice "young female with energetic and friendly voice"

python main.py "This is a test" --voice "deep authoritative male voice, calm and clear"
```

## Output

Generated audio saved to `results/` folder as WAV file.

## Requirements

- CUDA GPU with 8GB+ VRAM (T4 works)
- COU works too
- Python 3.8+

## Install

```bash
pip install -r requirements.txt
```

## Model

Downloads automatically on first run (~3.5GB):
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`

## Features

- Voice design via text prompt (no reference audio needed)
- 1.7B parameter model
