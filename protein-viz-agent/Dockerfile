FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY protein-viz-agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY protein-viz-agent/server.py .
COPY protein-viz-agent/templates/ ./templates/
COPY logo/ ./logo/
RUN mkdir -p /app/structures

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
