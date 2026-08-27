# GPU variant of ../Dockerfile. Same tooling, same user, same lifecycle scripts —
# only the base image and the CUDA-enabled Python wheels differ.
#
# KEEP IN SYNC with ../Dockerfile when you change tooling or the user setup.
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ARG TZ
ENV TZ="$TZ"
ENV DEBIAN_FRONTEND=noninteractive

ARG CLAUDE_CODE_VERSION=latest

# The CUDA images are plain Ubuntu with no Python, so install 3.11 from deadsnakes
# (jammy ships 3.10) and make it the default `python3`.
RUN apt-get update && \
  apt-get install -y --no-install-recommends software-properties-common curl ca-certificates gnupg && \
  add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Node.js 22.x (LTS) from NodeSource.
# `-t nodistro` forces the NodeSource suite over the distro's own `nodejs`
# package, which does not bundle npm; the version check fails the layer here
# rather than at the `npm install -g claude-code` step further down.
RUN apt-get update && \
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
  apt-get install -y -t nodistro nodejs && \
  node --version && npm --version && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# Development tools, in one layer. (See ../Dockerfile — same list.)
RUN apt-get update && apt-get install -y --no-install-recommends \
  less \
  git \
  openssh-client \
  procps \
  sudo \
  fzf \
  zsh \
  man-db \
  unzip \
  gnupg2 \
  iptables \
  iproute2 \
  dnsutils \
  jq \
  nano \
  vim \
  wget \
  build-essential \
  gfortran \
  libopenblas-dev \
  liblapack-dev \
  pkg-config \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools

# Create node user (UID 1000 to match typical dev environments)
ARG USERNAME=node
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
  useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME && \
  echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
  chmod 0440 /etc/sudoers.d/$USERNAME

RUN mkdir -p /usr/local/share/npm-global && \
  chown -R node:node /usr/local/share

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME /commandhistory

ENV DEVCONTAINER=true

RUN mkdir -p /workspace /home/node/.claude && \
  chown -R node:node /workspace /home/node/.claude

WORKDIR /workspace

ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
  wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

USER node

ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin
ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano

ARG ZSH_IN_DOCKER_VERSION=1.2.0
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
  -p git \
  -p fzf \
  -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
  -a "source /usr/share/doc/fzf/examples/completion.zsh" \
  -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  -x

RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

USER root

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
  mv /root/.local/bin/uv /usr/local/bin/uv

# Container-level tooling (same list as the CPU image).
COPY .devcontainer/requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt && \
  rm /tmp/requirements.txt

# CUDA-enabled wheels. Installed AFTER requirements.txt so they overwrite any
# CPU-only build of the same package. Edit gpu-requirements.txt to pick your
# framework — it ships empty on purpose, since installing both torch and jax
# CUDA stacks pulls several GB you probably do not want.
COPY .devcontainer/gpu/gpu-requirements.txt /tmp/gpu-requirements.txt
RUN if [ -s /tmp/gpu-requirements.txt ] && grep -qv '^\s*\(#.*\)\?$' /tmp/gpu-requirements.txt; then \
      uv pip install --system -r /tmp/gpu-requirements.txt; \
    else \
      echo "gpu-requirements.txt is empty - no CUDA wheels installed."; \
    fi && \
    rm /tmp/gpu-requirements.txt

USER node

RUN echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc && \
  echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc && \
  echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> ~/.bashrc && \
  echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> ~/.zshrc

USER root
RUN mkdir -p /home/node/.config/matplotlib
COPY .devcontainer/matplotlibrc /home/node/.config/matplotlib/matplotlibrc
RUN chown -R node:node /home/node/.config

COPY .devcontainer/post-create.sh /usr/local/bin/post-create.sh
RUN chmod +x /usr/local/bin/post-create.sh

USER node
