#!/bin/bash

# The official installer (https://ollama.com/download) runs Ollama automatically
# as a background service — no need to call `ollama serve` manually.

ollama pull llama3.2:latest
echo "Ollama is ready at http://localhost:11434"