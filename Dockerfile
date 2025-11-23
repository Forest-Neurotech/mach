# syntax=docker/dockerfile:1

# Development environment for mach-beamform
# Provides CUDA compilation without requiring local CUDA installation

ARG CUDA_VERSION=12.6.3
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    cmake \
    ninja-build \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (will automatically install Python when needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda \
    PATH="${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Silence warning about not being able to use hard links with cache mount
ENV UV_LINK_MODE=copy

# Set working directory
WORKDIR /workspace

# Copy dependency files first for better layer caching
# Dependencies only rebuild when these files change
COPY pyproject.toml uv.lock ./

# Install dependencies with cache mount
# This layer is cached and reused when only source code changes
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy the rest of the project
# Source code changes won't trigger dependency reinstall
COPY . .

# Install the project itself (fast, no dependency downloads)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# OCI labels
LABEL org.opencontainers.image.title="mach-beamform-dev" \
      org.opencontainers.image.description="Development environment for ultrafast GPU-accelerated beamforming" \
      org.opencontainers.image.source="https://github.com/Forest-Neurotech/mach" \
      org.opencontainers.image.vendor="Forest Neurotech"

# Default command: interactive bash shell
CMD ["/bin/bash"]
