# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml README.md LICENSE requirements.txt ./
COPY src ./src

RUN python -m pip install --prefix=/install --no-compile -r requirements.txt \
    && python -m pip install --prefix=/install --no-compile --no-deps .

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 10001 --system forecast \
    && useradd --uid 10001 --system --gid forecast --create-home forecast

COPY --from=builder /install /usr/local

WORKDIR /workspace
USER forecast

ENTRYPOINT ["fpl-forecast"]
CMD ["--help"]
