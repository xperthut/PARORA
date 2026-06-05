#!/bin/bash

set -e

# Run from project root so the Dockerfile can access both protein-viz-agent/ and logo/
cd "$(dirname "$0")"
mkdir -p protein-viz-agent/structures

docker rm -f $(docker ps -a -q --filter ancestor=parora) 2>/dev/null || true
docker rmi -f parora 2>/dev/null || true
docker build -t parora -f protein-viz-agent/Dockerfile .
docker run -p 8000:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v "$(pwd)/protein-viz-agent/structures:/app/structures" \
  parora
