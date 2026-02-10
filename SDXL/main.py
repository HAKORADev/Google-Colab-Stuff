import os
import sys
import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from PIL import Image
import warnings
import time

warnings.filterwarnings("ignore")

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
MODEL_CACHE = "model"
RESULTS_DIR = "results"

def setup_directories():
    Path(MODEL_CACHE).mkdir(exist_ok=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True)

def load_pipeline():
    print("Loading SDXL pipeline...")

    vae = AutoencoderKL.from_pretrained(
        VAE_ID,
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        cache_dir=MODEL_CACHE
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True
    )

    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()

    print("Pipeline loaded successfully!")
    return pipe

def generate_image(pipe, prompt, output_path):
    print(f"Generating: {prompt[:50]}...")

    image = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted, ugly, bad anatomy, watermark, signature, text, cropped, worst quality",
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=torch.Generator(device="cuda").manual_seed(torch.randint(0, 2**32, (1,)).item())
    ).images[0]

    image.save(output_path, quality=95, optimize=True)
    print(f"Saved: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="SDXL Image Generator")
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

    pipe = load_pipeline()

    timestamp = str(int(time.time() * 1000))[-6:]
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
    filename = f"{safe_prompt}_{timestamp}.png"
    output_path = os.path.join(RESULTS_DIR, filename)

    try:
        generate_image(pipe, prompt, output_path)
        print(f"\nDone! Output: {output_path}")
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()        negative_prompt="blurry, low quality, distorted, ugly, bad anatomy, watermark, signature, text, cropped, worst quality",
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=torch.Generator(device="cuda").manual_seed(torch.randint(0, 2**32, (1,)).item())
    ).images[0]

    image.save(output_path, quality=95, optimize=True)
    print(f"Saved: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="SDXL Image Generator")
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

    pipe = load_pipeline()

    timestamp = str(int(time.time() * 1000))[-6:]
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
    filename = f"{safe_prompt}_{timestamp}.png"
    output_path = os.path.join(RESULTS_DIR, filename)

    try:
        generate_image(pipe, prompt, output_path)
        print(f"\nDone! Output: {output_path}")
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main(        cache_dir=MODEL_CACHE
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True
    )

    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()

    print("Pipeline loaded successfully!")
    return pipe

def generate_image(pipe, prompt, output_path):
    print(f"Generating: {prompt[:50]}...")

    image = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted, ugly, bad anatomy, watermark, signature, text, cropped, worst quality",
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=torch.Generator(device="cuda").manual_seed(torch.randint(0, 2**32, (1,)).item())
    ).images[0]

    image.save(output_path, quality=95, optimize=True)
    print(f"Saved: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="SDXL Image Generator")
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

    pipe = load_pipeline()

    timestamp = str(int(time.time() * 1000))[-6:]
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
    filename = f"{safe_prompt}_{timestamp}.png"
    output_path = os.path.join(RESULTS_DIR, filename)

    try:
        generate_image(pipe, prompt, output_path)
        print(f"\nDone! Output: {output_path}")
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True
    )

    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()

    print("Pipeline loaded successfully!")
    return pipe

def generate_image(pipe, prompt, output_path):
    print(f"Generating: {prompt[:50]}...")

    image = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted, ugly, bad anatomy, watermark, signature, text, cropped, worst quality",
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=torch.Generator(device="cuda").manual_seed(torch.randint(0, 2**32, (1,)).item())
    ).images[0]

    image.save(output_path, quality=95, optimize=True)
    print(f"Saved: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="SDXL Image Generator")
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

    pipe = load_pipeline()

    timestamp = str(int(torch.cuda.Event(enable_timing=False).elapsed_time(torch.cuda.Event(enable_timing=False))))[-6:]
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
    filename = f"{safe_prompt}_{timestamp}.png"
    output_path = os.path.join(RESULTS_DIR, filename)

    try:
        generate_image(pipe, prompt, output_path)
        print(f"\nDone! Output: {output_path}")
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
