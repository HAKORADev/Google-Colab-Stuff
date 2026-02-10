import os
import sys
import argparse
import subprocess
from pathlib import Path

MODEL_DIR = "models/qwen_tts_voice_design"
RESULTS_DIR = "results"

def ensure_model():
    try:
        from qwen_tts import Qwen3TTSModel
        import torch

        os.makedirs(MODEL_DIR, exist_ok=True)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print("Loading Qwen3-TTS VoiceDesign model...")
        print(f"Device: {device}")

        if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
            print("Loading from local cache...")
            model = Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map=device, dtype=dtype)
        else:
            print("Downloading from HuggingFace (this may take 5-10 minutes)...")
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                device_map=device,
                dtype=dtype
            )

        print("Model loaded!")
        return model

    except ImportError:
        print("Error: qwen-tts not installed")
        print("Install: pip install qwen-tts")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def generate_speech(model, text, voice_prompt, output_path):
    try:
        import soundfile as sf

        print(f"Generating speech...")
        print(f"Text: {text[:50]}...")
        print(f"Voice: {voice_prompt[:50]}...")

        wavs, sr = model.generate_voice_design(
            text=text,
            language="English",
            instruct=voice_prompt
        )

        sf.write(output_path, wavs[0], sr)
        print(f"Saved: {output_path}")
        return True

    except Exception as e:
        print(f"Generation error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Voice Design")
    parser.add_argument("script", nargs="+", help="Text to synthesize")
    parser.add_argument("--voice", type=str, default="a clear adult male with professional tone", 
                        help="Voice description/prompt")
    args = parser.parse_args()

    text = " ".join(args.script)
    if not text:
        print("Error: No text provided")
        sys.exit(1)

    voice_prompt = args.voice

    os.makedirs(RESULTS_DIR, exist_ok=True)

    import torch
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("Warning: CUDA not available, using CPU (slow)")

    model = ensure_model()

    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_text = "".join(c if c.isalnum() else "_" for c in text[:20])
    output_path = os.path.join(RESULTS_DIR, f"tts_{safe_text}_{timestamp}.wav")

    success = generate_speech(model, text, voice_prompt, output_path)

    if success:
        print(f"\nDone! Output: {output_path}")
        try:
            from IPython.display import Audio, display
            display(Audio(output_path))
        except:
            pass
    else:
        print("\nFailed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
