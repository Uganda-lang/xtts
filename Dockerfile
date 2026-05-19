FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y \
    git curl gcc g++ build-essential \
    libsndfile1 ffmpeg wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

RUN git clone https://github.com/coqui-ai/TTS /tmp/TTS \
    && cd /tmp/TTS \
    && uv pip install --system -e ".[all]" \
    && rm -rf /tmp/TTS/.git

RUN uv pip install --system "transformers==4.37.2"

RUN pip install --no-cache-dir \
    torch==2.2.0+cu121 torchaudio==2.2.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN uv pip install --system runpod huggingface_hub

WORKDIR /app
COPY download_model.py .
COPY handler.py .
COPY ref ./ref

EXPOSE 8080

CMD ["bash", "-c", "python3 download_model.py && python3 handler.py"]