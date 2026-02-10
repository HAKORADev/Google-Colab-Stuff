import os
import sys
import argparse
import torch
from pathlib import Path
from diffusers import FluxPipeline
from PIL import Image
import warnings
import time

warnings.filterwarnings("ignore")

MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_CACHE = "model"
RESULTS_DIR = "results"

def setup_directories():
    Path(MODEL_CACHE).mkdir(exist_ok=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True)

def load_pipeline():
    print("Loading FLUX.1-dev pipeline...")
    print("This may take 2-5 minutes on first run (downloading 23GB model)")

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE
    )

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    print("Pipeline loaded successfully!")
    return pipe

def generate_image(pipe, prompt, output_path, seed=None):
    print(f"Generating: {prompt[:60]}...")

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt="low quality, blurry, distorted, bad anatomy, watermark, signature, text, cropped, worst quality",
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=3.5,
        generator=generator
    ).images[0]

    image.save(output_path, quality=95, optimize=True)
    print(f"Saved: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="FLUX.1-dev Image Generator")
    parser.add_argument("prompt", nargs="+", help="Image generation prompt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    prompt = " ".join(args.prompt)
    if not prompt:
        print("Error: No prompt provided")
        sys.exit(1)

    setup_directories()

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)

    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print("Model: FLUX.1-dev (SOTA 2024)")

    pipe = load_pipeline()

    timestamp = str(int(time.time() * 1000))[-6:]
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
    filename = f"{safe_prompt}_{timestamp}.png"
    output_path = os.path.join(RESULTS_DIR, filename)

    try:
        generate_image(pipe, prompt, output_path, args.seed)
        print(f"\nDone! Output: {output_path}")
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
