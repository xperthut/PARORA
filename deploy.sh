#!/bin/bash

set -e

cd protein-viz-agent
mkdir -p structures

docker rm -f $(docker ps -a -q --filter ancestor=protein-viz-agent) 2>/dev/null || true
docker rmi -f protein-viz-agent 2>/dev/null || true
docker build -t protein-viz-agent .
docker run -p 8501:8501 \
  -v "$(pwd)/structures:/app/structures" \
  protein-viz-agent