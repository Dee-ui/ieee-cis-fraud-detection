# The prediction service.
#
# Built on python:3.11-slim: the same Python as development, without the
# several hundred megabytes of build tools the full image carries.

FROM python:3.11-slim

# LightGBM needs the OpenMP runtime, which the slim image leaves out. Without
# it, importing lightgbm fails with a missing libgomp.so.1 that gives no hint
# about what to install.
#
# Cleaning the apt lists in the same RUN matters: each instruction becomes a
# layer, so deleting them in a later instruction would leave them in the
# earlier layer and the image would still carry the weight.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as a non-root user with id 1000.
# Creating that user here means the image behaves the same locally as it
# does on Spaces, instead of working on your machine and failing there.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    ARTIFACT_CACHE_DIR=/home/user/.cache/artifacts

WORKDIR $HOME/app

# Dependencies before source. Editing a Python file then rebuilds only the
# layers below this point; the pip install is reused from cache.
COPY --chown=user requirements-serve.txt .
RUN pip install --no-cache-dir --user -r requirements-serve.txt

# Only what the service needs. No data, no notebooks, no tests, no reports.
COPY --chown=user config/ ./config/
COPY --chown=user src/ ./src/

# Spaces expects 7860; Render injects its own PORT (default 10000).
# Falling back to 7860 keeps local `docker run` unchanged.
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request, json; \
        port = os.environ.get('PORT', '7860'); \
        r = json.loads(urllib.request.urlopen(f'http://localhost:{port}/health').read()); \
        exit(0 if r['model_loaded'] else 1)"

# Shell form so ${PORT} is expanded at container start. Render injects its
# own PORT (default 10000); the fallback keeps local `docker run` and
# Hugging Face Spaces (which expects 7860) working unchanged.
CMD uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT:-7860}
