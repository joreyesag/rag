FROM mcr.microsoft.com/devcontainers/python:3.10-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    zsh curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.cargo/bin/uv /usr/local/bin/ \
    && mv /root/.cargo/bin/uvx /usr/local/bin/

RUN cat >> /home/vscode/.zshrc <<'EOF'
PROMPT="%F{cyan}┌[%F{red}%n%F{cyan}]-[%F{yellow}%~%F{cyan}]%f
%F{cyan}└> %f"

if [ -d "/workspaces/$(basename $PWD)/.venv" ]; then
    source /workspaces/$(basename $PWD)/.venv/bin/activate
fi
EOF

RUN chown vscode:vscode /home/vscode/.zshrc \
    && chsh -s /usr/bin/zsh vscode

CMD ["/usr/bin/zsh"]FROM mcr.microsoft.com/devcontainers/python:3.10-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    zsh curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.cargo/bin/uv /usr/local/bin/ \
    && mv /root/.cargo/bin/uvx /usr/local/bin/

RUN cat >> /home/vscode/.zshrc <<'EOF'
PROMPT="%F{cyan}┌[%F{red}%n%F{cyan}]-[%F{yellow}%~%F{cyan}]%f
%F{cyan}└> %f"

if [ -d "/workspaces/$(basename $PWD)/.venv" ]; then
    source /workspaces/$(basename $PWD)/.venv/bin/activate
fi
EOF

RUN chown vscode:vscode /home/vscode/.zshrc \
    && chsh -s /usr/bin/zsh vscode

CMD ["/usr/bin/zsh"]