# =============================================================================
# Stage 1: Build stage - Install dependencies
# =============================================================================
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bullseye AS builder

WORKDIR /app

# Install build dependencies (git is needed for git-based dependencies in pyproject.toml)
RUN apt-get update -y \
  && apt-get install -y --no-install-recommends git \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN pip install --no-cache-dir poetry

# Copy only dependency files first for better layer caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies (without dev dependencies)
RUN poetry install --no-interaction --no-ansi --no-root --only main

# Copy application code and install the project
COPY . .
RUN poetry install --no-interaction --no-ansi --only main

# =============================================================================
# Stage 2: Runtime stage - Minimal image with only what's needed
# =============================================================================
FROM python:${PYTHON_VERSION}-slim-bullseye AS runtime

ARG HUGO_VERSION=0.121.2
ARG TARGETARCH

WORKDIR /app

# Install only runtime dependencies
# - git: required for cloning Hugo website repos during ingest operation
# - wget: required for downloading Hugo
# - ca-certificates: required for HTTPS connections
RUN apt-get update -y \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install Hugo (required for ingest operation to build Hugo websites)
# Note: We download the pre-built binary directly, no need for Go
RUN wget -q https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-${TARGETARCH}.tar.gz \
  && tar -C /usr/local/bin -xzf hugo_extended_${HUGO_VERSION}_linux-${TARGETARCH}.tar.gz hugo \
  && rm hugo_extended_${HUGO_VERSION}_linux-${TARGETARCH}.tar.gz \
  && hugo version

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser . .

# Set environment variables
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Run main.py when the container launches
CMD ["python", "main.py"]
