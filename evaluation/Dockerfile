# Training and evaluation in one image: slime's environment (SGLang v0.5.15.post1,
# CUDA 12.9, its own SGLang patches) plus the Harbor CLI.
#
#   docker build -t terl-eval -f evaluation/Dockerfile .
#
# Pin the base to a digest for reproducible runs; :latest moves.
ARG SLIME_IMAGE=slimerl/slime:latest
FROM ${SLIME_IMAGE}

# uv installs to ~/.local/bin, which is not on PATH by default.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Harbor gets its own isolated venv, so it cannot perturb slime's pinned
# torch / sglang / sgl-kernel versions.
RUN uv tool install harbor

# Docker CLI only — no daemon. Harbor talks to the HOST daemon through the
# socket mounted at run time (see README), creating sibling containers.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      -o /etc/apt/keyrings/docker.asc \
 && chmod a+r /etc/apt/keyrings/docker.asc \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
      > /etc/apt/sources.list.d/docker.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
 && rm -rf /var/lib/apt/lists/*

RUN harbor --version && python -c "import sglang; print('sglang', sglang.__version__)"

WORKDIR /workspace
