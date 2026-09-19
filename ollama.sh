#!/bin/bash

# The official installer (https://ollama.com/download) runs Ollama automatically
# as a background service — no need to call `ollama serve` manually.

ollama pull qwen2.5:7b   # used by server.py / app.py (the Docker default)
ollama pull llama3.2     # used by app_lite.py
echo "Ollama is ready at http://localhost:11434"