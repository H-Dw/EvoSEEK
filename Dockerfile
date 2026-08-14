FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE THIRD_PARTY.md ./
COPY requirements ./requirements
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install -e ".[dev]"
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY data/demo ./data/demo
RUN mkdir -p artifacts/runs models data/raw data/interim data/processed

CMD ["python", "scripts/run_demo.py", "--rounds", "1", "--budget", "4"]

