# ---- builder stage --------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies only (caching layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install .

# Copy application source and install the project itself
COPY . .
RUN pip install --no-cache-dir --prefix=/install .


# ---- runtime stage --------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL maintainer="agent-runner team"
LABEL description="Enterprise Agent Runner — production image"

# Create non-root user
RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid agent --shell /bin/bash --create-home agent

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source (for the entry-point module resolution)
WORKDIR /app
COPY --chown=agent:agent . .

# Runtime configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGENT_RUNNER_HOST=0.0.0.0 \
    AGENT_RUNNER_PORT=8811

EXPOSE 8811

# Health check — hits the liveness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8811/health/live')" || exit 1

# Drop to non-root
USER agent

# Use exec form so Python receives SIGTERM directly
CMD ["python", "-m", "main", "--serve-approvals"]
