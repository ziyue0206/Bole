# syntax=docker/dockerfile:1
# Thin Bole runtime layer on the official OpenAI development environment.
# Build context: checksum-verified codex-tools directory prepared by bootstrap.
ARG CODEX_BASE=ghcr.io/openai/codex-universal:latest
FROM ${CODEX_BASE}
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/bole-tools/codex-resources /home/bole/.codex /workspace \
    && cp /etc/profile /opt/bole-upstream-profile
COPY --chmod=0644 profile.sh /etc/profile
COPY --chmod=0755 codex /opt/bole-tools/codex
COPY --chmod=0755 bwrap /opt/bole-tools/codex-resources/bwrap
ARG CODEX_VERSION
LABEL io.bole.codex.version=${CODEX_VERSION}
ENV HOME=/home/bole CODEX_HOME=/home/bole/.codex
ENV PATH=/opt/bole-tools:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WORKDIR /workspace
USER 1000:1000
ENTRYPOINT ["/opt/bole-tools/codex"]
CMD ["--version"]
