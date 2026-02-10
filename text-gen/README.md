# GPT-OSS 20B Chatbot

chatbot running GPT-OSS 20B Q4_K_M on T4 GPU.

## Usage

```bash
pythongptoss20b.py
```

First run downloads ~11GB model automatically.

## Commands

- `quit`, `exit`, `/bye` — Exit
- `/clear` — Clear conversation history

## Features

- **32k context window** with 95% sliding window
- **Thinking model** — shows reasoning process (optional)
- **Markdown output** — Formatted responses
- **GPU accelerated** — 35 layers offloaded to T4

## Requirements

- T4 GPU (16GB VRAM)
- ~12GB disk space
- llama.cpp installed

## Install requirements

```bash
pip install -r requirements.txt
```
---
Have fun!
