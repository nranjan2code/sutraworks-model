# Multi-stage Docker build for SutraWorks HTTP API server
# Based on advanced AI framework with production security hardening

FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -r -s /bin/false -m sutraworks

# Set up build directory
WORKDIR /usr/src/app

# Copy workspace configuration
COPY Cargo.toml .
COPY Cargo.lock .
COPY crates/ crates/

# Build the sutra-server binary
RUN cargo build --release --bin sutra-server

# Runtime stage - minimal image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false -m sutraworks

# Copy binary from builder
COPY --from=builder /usr/src/app/target/release/sutra-server /usr/local/bin/sutra-server

# Set binary permissions
RUN chmod +x /usr/local/bin/sutra-server

# Switch to non-root user
USER sutraworks

# Expose port for HTTP server
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8003/health || exit 1

# Default command - serve with RWKV model
CMD ["sutra-server", "serve", "--port", "8003", "--model", "rwkv", "--warmup"]

# Labels for container metadata
LABEL org.opencontainers.image.title="SutraWorks HTTP Server"
LABEL org.opencontainers.image.description="Production HTTP API for SutraWorks AI models"
LABEL org.opencontainers.image.vendor="SutraWorks"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/nranjan2code/sutraworks-model"
