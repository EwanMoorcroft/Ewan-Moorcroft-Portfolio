# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src

RUN python -m pip install --prefix=/install --no-compile -r requirements.txt \
    && python -m pip install --prefix=/install --no-compile --no-deps .

FROM python:3.11-slim AS runtime

ENV MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 10001 --system analyst \
    && useradd --uid 10001 --system --gid analyst --create-home analyst

COPY --from=builder /install /usr/local

WORKDIR /workspace
USER analyst

ENTRYPOINT ["liverpool-access"]
CMD ["--help"]
