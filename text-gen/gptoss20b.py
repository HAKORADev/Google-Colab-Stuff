import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

MODEL_URL = "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q4_K_M.gguf"
MODEL_NAME = "gpt-oss-20b-Q4_K_M.gguf"
MODEL_DIR = "models"
CONTEXT_SIZE = 128000
MAX_CONTEXT_RATIO = 0.95
TEMPERATURE = 0.7

def ensure_llama_cpp():
    try:
        import llama_cpp
        print(f"llama-cpp-python found: {llama_cpp.__version__}")
        return True
    except ImportError:
        return False

def install_llama_cpp():
    print("llama.cpp not found. Installing...")
    print("Run: pip install llama-cpp-python")
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
        print(f"wget https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q4_K_M.gguf -O {model_path}")
        sys.exit(1)

class ChatSession:
    def __init__(self, model_path):
        from llama_cpp import Llama
        self.model_path = model_path
        self.messages = []
        self.system_prompt = "You are a helpful AI assistant. Think step by step and provide detailed, accurate responses."

        print("Loading model into VRAM...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=CONTEXT_SIZE,
            n_gpu_layers=35,
            verbose=False,
            n_batch=512,
        )
        print("Model loaded!")

    def estimate_tokens(self, text):
        return len(text) // 4

    def get_context_window(self):
        total_tokens = sum(self.estimate_tokens(m["content"]) for m in self.messages)

        while total_tokens > int(CONTEXT_SIZE * MAX_CONTEXT_RATIO) and len(self.messages) > 1:
            removed = self.messages.pop(0)
            total_tokens -= self.estimate_tokens(removed["content"])

        return self.messages

    def format_prompt(self):
        messages = self.get_context_window()
        lines = []

        lines.append("<|im_start|>system")
        lines.append(self.system_prompt)
        lines.append("<|im_end|>")

        for msg in messages:
            lines.append("<|im_start|>" + msg["role"])
            lines.append(msg["content"])
            lines.append("<|im_end|>")

        lines.append("<|im_start|>assistant")

        return "\n".join(lines)

    def generate(self, user_input, show_thinking=False):
        self.messages.append({"role": "user", "content": user_input})

        prompt = self.format_prompt()

        try:
            output = self.llm(
                prompt,
                max_tokens=2048,
                temperature=TEMPERATURE,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["<|im_end|>", "<|im_start|>user"],
            )

            text = output["choices"][0]["text"].strip()

            if "<|thinking|>" in text:
                thinking_start = text.find("<|thinking|>") + len("<|thinking|>")
                thinking_end = text.find("<|/thinking|>")

                if thinking_end > thinking_start:
                    thinking = text[thinking_start:thinking_end].strip()
                    response = text[thinking_end + len("<|/thinking|>"):].strip()

                    if show_thinking:
                        print("\n🤔 Thinking: " + thinking[:200] + "...")

                    text = response

            self.messages.append({"role": "assistant", "content": text})
            return text

        except Exception as e:
            return "[Error: " + str(e) + "]"

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

    print("\nInitializing...")
    print("Context: " + str(CONTEXT_SIZE) + " tokens (sliding " + str(int(MAX_CONTEXT_RATIO*100)) + "%)")
    print("Temperature: " + str(TEMPERATURE))
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
                print("🤔 Thinking visibility: " + str(show_thinking))
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
