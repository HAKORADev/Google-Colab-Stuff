import os
import sys
import argparse
import subprocess
from pathlib import Path

MODEL_URL = "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q4_K_M.gguf"
MODEL_NAME = "gpt-oss-20b-Q4_K_M.gguf"
MODEL_DIR = "models"
CONTEXT_SIZE = 32768
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
        print(f"wget {MODEL_URL} -O {model_path}")
        sys.exit(1)

class ChatSession:
    def __init__(self, model_path):
        from llama_cpp import Llama
        self.model_path = model_path
        self.messages = []
        self.system_prompt = "You are a helpful AI assistant."

        print("Loading model (this may take 1-2 minutes)...")
        print("GPU layers: 35/49 on T4")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=CONTEXT_SIZE,
            n_gpu_layers=35,
            verbose=True,
            n_batch=512,
        )
        print("Model ready!")

    def estimate_tokens(self, text):
        return len(text) // 4

    def get_context_window(self):
        total_tokens = sum(self.estimate_tokens(m["content"]) for m in self.messages)

        while total_tokens > int(CONTEXT_SIZE * MAX_CONTEXT_RATIO) and len(self.messages) > 1:
            removed = self.messages.pop(0)
            total_tokens -= self.estimate_tokens(removed["content"])

        return self.messages

    def format_prompt(self):
        parts = []
        parts.append("<|start|>system<|message|>" + self.system_prompt + "<|end|>")

        for msg in self.get_context_window():
            if msg["role"] == "user":
                parts.append("<|start|>user<|message|>" + msg["content"] + "<|end|>")
            elif msg["role"] == "assistant":
                parts.append("<|start|>assistant<|channel|>final<|message|>" + msg["content"] + "<|end|>")

        parts.append("<|start|>assistant<|channel|>final<|message|>")
        return "".join(parts)

    def generate_stream(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        prompt = self.format_prompt()

        print("🤖 Assistant: ", end="", flush=True)

        response_text = ""
        try:
            stream = self.llm(
                prompt,
                max_tokens=1024,
                temperature=TEMPERATURE,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["<|end|>", "<|return|>", "<|start|>user", "<|start|>system"],
                stream=True,
            )

            for output in stream:
                token = output["choices"][0]["text"]
                response_text += token
                print(token, end="", flush=True)

            print()

            clean_text = response_text.replace("<|end|>", "").replace("<|return|>", "").strip()
            self.messages.append({"role": "assistant", "content": clean_text})
            return clean_text

        except Exception as e:
            print(f"\n[Error: {e}]")
            return "[Generation failed]"

def main():
    parser = argparse.ArgumentParser(description="Local GPT-OSS 20B Chat")
    parser.add_argument("--model-path", type=str, help="Path to GGUF model")
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

    print(f"\nConfig:")
    print(f"  Context: {CONTEXT_SIZE} tokens (sliding {int(MAX_CONTEXT_RATIO*100)}%)")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Note: First response may take 30-60s on T4")
    print("\nCommands: /clear, /bye, quit, exit")
    print("=" * 50)

    chat = ChatSession(model_path)

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

            chat.generate_stream(user_input)

        except KeyboardInterrupt:
            print("\n👋 Interrupted")
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
