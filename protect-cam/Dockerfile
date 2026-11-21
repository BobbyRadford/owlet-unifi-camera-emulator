# The Owlet Cam as a native UniFi camera: reolink-unifi-protect (pinned) for the
# Protect camera protocol, plus owlet_cam for detection and talk-back.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg netcat-openbsd git build-essential libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

ARG PROXY_COMMIT=5972d89332737c0627ad451f7898d3f00f4c0bdc
RUN git clone -q https://github.com/woutersteegers/reolink-unifi-protect.git /src/proxy \
    && git -C /src/proxy checkout -q $PROXY_COMMIT

WORKDIR /app
RUN pip install --no-cache-dir -r /src/proxy/requirements.txt \
        numpy onnxruntime ai-edge-litert pillow pytest
# The proxy's --model list stops at the G5 generation; the Owlet is a G6 Instant.
RUN cp -r /src/proxy/unifi /app/unifi \
    && sed -i 's/"Vision Pro",/"Vision Pro",\n            "UVC G6 Instant",/' /app/unifi/main.py \
    && grep -q '"UVC G6 Instant"' /app/unifi/main.py

COPY models /app/models
COPY owlet_cam /app/owlet_cam
COPY tests /app/tests

ENTRYPOINT ["python", "-m", "owlet_cam"]
