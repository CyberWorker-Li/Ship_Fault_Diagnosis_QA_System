FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# 先安装 CPU 版 torch + torchvision，避免 CUDA 依赖下载，并满足 transformers 的图像子模块依赖
RUN pip install --retries 5 --timeout 120 --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1+cpu \
    torchvision==0.20.1+cpu

RUN pip install --retries 5 --timeout 120 \
    streamlit \
    langchain \
    langchain-openai \
    langchain-text-splitters \
    sentence-transformers \
    faiss-cpu \
    rank-bm25 \
    jieba \
    networkx \
    requests \
    neo4j \
    pypdf \
    python-docx \
    numpy

COPY . /app

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]