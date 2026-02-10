import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

MODEL_URL = "https://huggingface.co/bartowski/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q4_K_M.gguf"
MODEL_NAME = "gpt-oss-20b-Q4_K_M.gguf"
MODEL_DIR = "models"
CONTEXT_SIZE = 128000
MAX_CONTEXT_RATIO = 0.95
TEMPERATURE = 0.7

def ensure_llama_cpp():
    try:
        subprocess.run(["llama-cli", "--version"], capture_output=True, check=True)
        return True
    except:
        return False

def install_llama_cpp():
    print("llama.cpp not found. Installing...")
    print("Run: pip install llama-cpp-python")
    print("Or build from source: https://github.com/ggerganov/llama.cpp")
    return False

def download_model():
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024**3)
        print(f"Model found: {model_path} ({size:.1f}GB)")
        return model_path

    print(f"Downloading {MODEL_NAME}...")
    print(f"URL: {MODEL_URL}")
    print("This is ~11GB and may take 10-20 minutes...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        subprocess.run([
            "wget", "-c", "--show-progress", 
            "-O", model_path,
            MODEL_URL
        ], check=True)
        print(f"Downloaded to {model_path}")
        return model_path
    except subprocess.CalledProcessError:
        print("Download failed. Try manual download:")
        print(f"wget {MODEL_URL} -O {model_path}")
        sys.exit(1)

class ChatSession:
    def __init__(self, model_path):
        self.model_path = model_path
        self.messages = []
        self.system_prompt = "You are a helpful AI assistant. Think step by step and provide detailed, accurate responses."

    def estimate_tokens(self, text):
        # Rough estimate: ~4 chars per token
        return len(text) // 4

    def get_context_window(self):
        total_tokens = sum(self.estimate_tokens(m["content"]) for m in self.messages)

        # Sliding window: keep max 95% of context
        while total_tokens > int(CONTEXT_SIZE * MAX_CONTEXT_RATIO) and len(self.messages) > 1:
            removed = self.messages.pop(0)
            total_tokens -= self.estimate_tokens(removed["content"])

        return self.messages

    def format_prompt(self):
        messages = self.get_context_window()

        prompt_parts = []

        # System
        prompt_parts.append(f"<|im_start|>system
{self.system_prompt}<|im_end|>")

        # History
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}
{content}<|im_end|>")

        # Current prompt
        prompt_parts.append("<|im_start|>assistant
")

        return "\n".join(prompt_parts)

    def generate(self, user_input, show_thinking=False):
        self.messages.append({"role": "user", "content": user_input})

        prompt = self.format_prompt()

        cmd = [
            "llama-cli",
            "-m", self.model_path,
            "-p", prompt,
            "-n", "2048",
            "--temp", str(TEMPERATURE),
            "--top-p", "0.9",
            "--top-k", "40",
            "--repeat-penalty", "1.1",
            "--ctx-size", str(CONTEXT_SIZE),
            "--batch-size", "512",
            "-ngl", "35",  # GPU layers for T4
            "--multiline-input",
            "--no-display-prompt"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            output = result.stdout.strip()

            # Parse thinking if present
            if "<|thinking|>" in output:
                thinking_start = output.find("<|thinking|>") + len("<|thinking|>")
                thinking_end = output.find("<|/thinking|>")

                if thinking_end > thinking_start:
                    thinking = output[thinking_start:thinking_end].strip()
                    response = output[thinking_end + len("<|/thinking|>"):].strip()

                    if show_thinking:
                        print(f"\n🤔 Thinking: {thinking[:200]}...")

                    output = response

            self.messages.append({"role": "assistant", "content": output})
            return output

        except subprocess.TimeoutExpired:
            return "[Generation timeout - model is slow on T4]"
        except Exception as e:
            return f"[Error: {e}]"

def main():
    parser = argparse.ArgumentParser(description="Local GPT-OSS 20B Chat")
    parser.add_argument("--show-thinking", action="store_true", help="Show model thinking process")
    parser.add_argument("--model-path", type=str, help="Path to GGUF model (auto-download if not set)")
    args = parser.parse_args()

    print("🤖 Local GPT-OSS 20B Chatbot")
    print("=" * 50)

    if not ensure_llama_cpp():
        if not install_llama_cpp():
            return

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = download_model()

    print(f"\nLoading model: {model_path}")
    print(f"Context: {CONTEXT_SIZE} tokens (sliding {MAX_CONTEXT_RATIO*100:.0f}%)")
    print(f"Temperature: {TEMPERATURE}")
    print("\nType 'quit', 'exit', or '/bye' to exit")
    print("Type '/clear' to clear history")
    print("Type '/think' to toggle thinking visibility")
    print("=" * 50)

    chat = ChatSession(model_path)
    show_thinking = args.show_thinking

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "/bye"]:
                print("👋 Goodbye!")
                break
            elif user_input == "/clear":
                chat.messages = []
                print("🧹 History cleared")
                continue
            elif user_input == "/think":
                show_thinking = not show_thinking
                print(f"🤔 Thinking visibility: {show_thinking}")
                continue

            print("🤖 Assistant: ", end="", flush=True)
            response = chat.generate(user_input, show_thinking)
            print(response)

        except KeyboardInterrupt:
            print("\n👋 Interrupted")
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
