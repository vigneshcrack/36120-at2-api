# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install uv (matches the project's toolchain used throughout the assignment)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

# Copy dependency manifests first so Docker can cache the dependency layer
# separately from application code changes.
COPY pyproject.toml uv.lock* ./

# --extra-index-url is required here too, since the custom mykrml_25740287
# package is published to TestPyPI, not the main PyPI index.
RUN uv sync --no-dev --extra-index-url https://test.pypi.org/simple/ --index-strategy unsafe-best-match

# Now copy the application code and trained model artefacts.
COPY app/ app/
COPY models/ models/

EXPOSE 8000

# Render sets $PORT at runtime; default to 8000 for local `docker run`.
ENV PORT=8000
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
