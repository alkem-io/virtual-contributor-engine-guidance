# =============================================================================
# Stage 1: Build stage - install dependencies using Debian Python for distroless compatibility
# =============================================================================
FROM debian:bookworm-slim AS builder

# Install Python, pip, git and build essentials in a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry in a separate venv to avoid PEP 668 issues
RUN python3 -m venv /opt/poetry && \
    /opt/poetry/bin/pip install --no-cache-dir poetry && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Configure Poetry to create venv in project
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy only dependency files first (better layer caching)
COPY pyproject.toml poetry.lock* ./

# Install dependencies (without dev dependencies)
RUN poetry install --only main --no-root --no-ansi && \
    rm -rf $POETRY_CACHE_DIR && \
    # Remove pip + wheel (keep setuptools for runtime compatibility)
    /app/.venv/bin/pip uninstall -y pip wheel && \
    find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + && \
    # Trim non-runtime content from installed packages
    SITE_PACKAGES="$(/app/.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')" && \
    find "$SITE_PACKAGES" -type d \( \
        -name "tests" -o -name "test" -o \
        -name "docs" -o -name "doc" -o \
        -name "examples" -o -name "example" -o \
        -name "__pycache__" \
    \) -prune -exec rm -rf '{}' + && \
    find "$SITE_PACKAGES" -type f -name "*.pyc" -delete

# Copy application code (kept after dependency install for better caching)
COPY . /app

# Create files/dirs that the distroless runtime cannot create (no shell)
RUN touch /app/app.log

# =============================================================================
# Stage 2: Runtime stage - Google distroless Python image
# =============================================================================
# NOTE: This image does NOT include Git or Hugo. The 'ingest' operation 
# will NOT work. Use the 'runtime-full' target if you need ingest capability.
# =============================================================================
FROM gcr.io/distroless/python3-debian12:nonroot AS runtime

# Copy the prebuilt app (code + venv + log file) from the builder.
# Use UID/GID 1000 to stay compatible with typical host-mounted volumes.
COPY --from=builder --chown=1000:1000 /app /app

WORKDIR /app

# Set PATH to use the venv's python executable
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/.venv/lib/python3.11/site-packages:/app"

# Explicitly define the user
USER 1000:1000

# Run main.py using the venv python
ENTRYPOINT ["/app/.venv/bin/python", "main.py"]

# =============================================================================
# Stage 3: Full runtime with Git and Hugo (for ingest capability)
# =============================================================================
FROM debian:bookworm-slim AS runtime-full

ARG HUGO_VERSION=0.121.2
ARG TARGETARCH

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        git \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Hugo
ADD https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-${TARGETARCH}.tar.gz /tmp/hugo.tar.gz
RUN tar -C /usr/local/bin -xzf /tmp/hugo.tar.gz hugo \
  && rm /tmp/hugo.tar.gz \
  && hugo version

# Create non-root user
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser . .

# Create a writable log file for the appuser
RUN touch /app/app.log && chown appuser:appuser /app/app.log

# Set environment variables
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Run main.py when the container launches
ENTRYPOINT ["/app/.venv/bin/python", "main.py"]
