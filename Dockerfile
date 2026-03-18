FROM python:3.11-slim

# Dependências nativas C/C++ exigidas pelo OpenCV no Linux
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar o gerenciador super-rápido UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Hugging Face exige que o container rode usando um usuário não-root (user ID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:${PATH}"

# Definir a pasta do container
WORKDIR /home/user/app

# Copiar os arquivos e sincronizar as dependências usando o UV
COPY --chown=user:user . .
RUN uv sync --frozen

# Expor a porta que o Hugging Face Spaces injeta por padrão
EXPOSE 7860

# Iniciar o servidor FastHTML através do UV
CMD ["uv", "run", "app.py"]
