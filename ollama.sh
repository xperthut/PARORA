#!/bin/bash

# The official installer (https://ollama.com/download) runs Ollama automatically
# as a background service — no need to call `ollama serve` manually.

ollama pull qwen2.5:7b
echo "Ollama is ready at http://localhost:11434"