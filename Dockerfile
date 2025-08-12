# Embodied AI Benchmark++ Production Dockerfile
# Multi-stage build for optimized production deployment

# Stage 1: Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install build && \
    pip install .

# Stage 2: Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    BENCHMARK_ENV=production \
    BENCHMARK_LOG_LEVEL=INFO

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    # Essential runtime libraries
    libgomp1 \
    # Networking tools for health checks
    curl \
    # Process management
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r benchmark && \
    useradd -r -g benchmark -d /app -s /bin/bash benchmark

# Create application directory
WORKDIR /app

# Copy application code
COPY --chown=benchmark:benchmark src/ ./src/
COPY --chown=benchmark:benchmark *.py ./
COPY --chown=benchmark:benchmark README.md LICENSE ./

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/cache && \
    chown -R benchmark:benchmark /app

# Switch to non-root user
USER benchmark

# Expose port for API server
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); import embodied_ai_benchmark; print('Health check passed')" || exit 1

# Default command
CMD ["python", "-m", "src.embodied_ai_benchmark.api.app"]

# Labels for metadata
LABEL org.opencontainers.image.title="Embodied AI Benchmark++" \
      org.opencontainers.image.description="Comprehensive evaluation suite for embodied AI systems" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT"