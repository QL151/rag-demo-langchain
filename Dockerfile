FROM python:3.12-slim

# 框架版本本次使用Python 3.12验证
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt requirements-chroma.txt .
RUN pip install --no-cache-dir -r requirements-chroma.txt

COPY app ./app
COPY static ./static
COPY scripts ./scripts
COPY tests ./tests
COPY pytest.ini .
COPY data/sample ./data/sample

RUN mkdir -p data/uploads data/chroma data/traces \
    # 不用 root 跑服务：容器被突破时能少一分风险，也是上线的最低要求
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查：编排工具（compose / k8s）据此判断容器是否可用
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
