"""Read-only HTTP service for one configured FPL forecast deployment."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as ApiPath
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from .config import ProtocolConfig
from .contracts import sha256_bytes, validate_season_id
from .data import load_gameweeks
from .errors import ArtifactError, ForecastingError
from .features import build_forecast_frame
from .models import ARTIFACT_FORMAT, FittedRidge
from .operational import (
    COMPLETION_STATUS,
    ranked_prediction_payload,
    validate_deployment_model,
    validate_expected_as_of,
)
from .storage import (
    MAX_ARTIFACT_BYTES,
    ModelRegistration,
    OperationalStore,
    StorageError,
    read_bounded_bytes,
)

MAX_REQUEST_BODY_BYTES = 16_384
MAX_PLAYER_IDS = 100
MAX_STORED_PREDICTIONS = 500
_SEASON_PATTERN = r"^[0-9]{4}-[0-9]{2}$"


class PredictRequest(BaseModel):
    """Strict, bounded selector for the already-configured forecast batch."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    season_id: str = Field(min_length=7, max_length=7, pattern=_SEASON_PATTERN)
    as_of_gw: StrictInt = Field(ge=1, le=60)
    player_ids: list[StrictInt] = Field(min_length=1, max_length=MAX_PLAYER_IDS)

    @field_validator("season_id")
    @classmethod
    def canonical_season(cls, value: str) -> str:
        try:
            return validate_season_id(value)
        except ForecastingError as exc:
            raise ValueError("season_id is not a consecutive FPL season") from exc

    @field_validator("player_ids")
    @classmethod
    def bounded_unique_player_ids(cls, values: list[int]) -> list[int]:
        if any(value < 1 or value > 2_147_483_647 for value in values):
            raise ValueError("player_ids must contain positive 32-bit integers")
        if len(set(values)) != len(values):
            raise ValueError("player_ids must be unique")
        return values


@dataclass(frozen=True)
class ServiceRuntime:
    """Validated immutable runtime state; the store is opened read-only."""

    season_id: str
    expected_as_of_gw: int
    completion_status: str
    model: FittedRidge
    model_registration: ModelRegistration
    artifact_sha256: str
    forecast_frame: Any
    store: OperationalStore

    @classmethod
    def load(
        cls,
        *,
        artifact_path: str | Path,
        gameweek_dir: str | Path,
        database_path: str | Path,
        season_id: str,
        expected_as_of_gw: int,
        completion_status: str,
        protocol_path: str | Path | None = None,
    ) -> ServiceRuntime:
        season = validate_season_id(season_id)
        if completion_status != COMPLETION_STATUS:
            raise ArtifactError("Service requires an explicit completed snapshot declaration")
        protocol = ProtocolConfig.from_json(protocol_path) if protocol_path else ProtocolConfig()
        snapshots = load_gameweeks(
            gameweek_dir,
            minimum_adjacent_coverage=protocol.minimum_adjacent_player_coverage,
            expected_season_id=season,
        )
        as_of_gw = validate_expected_as_of(snapshots, expected_as_of_gw)
        frame = build_forecast_frame(
            snapshots,
            minimum_adjacent_coverage=protocol.minimum_adjacent_player_coverage,
        )
        artifact_bytes = read_bounded_bytes(
            artifact_path,
            max_bytes=MAX_ARTIFACT_BYTES,
            label="model artifact",
        )
        model = FittedRidge.from_json_bytes(artifact_bytes)
        validate_deployment_model(
            model,
            season_id=season,
            config_sha256=protocol.sha256(),
            as_of_gw=as_of_gw,
        )
        artifact_sha256 = sha256_bytes(artifact_bytes)
        registration = ModelRegistration(
            season_id=season,
            artifact_format=ARTIFACT_FORMAT,
            artifact_sha256=artifact_sha256,
            feature_schema_version=model.feature_schema_version,
            feature_schema_sha256=model.feature_schema_sha256,
            protocol_config_sha256=protocol.sha256(),
            trained_through_target_gw=model.trained_through_target_gw,
            training_rows=model.training_rows,
        )
        store = OperationalStore(database_path, read_only=True)
        store.require_registered_model(registration)
        return cls(
            season_id=season,
            expected_as_of_gw=as_of_gw,
            completion_status=completion_status,
            model=model,
            model_registration=registration,
            artifact_sha256=artifact_sha256,
            forecast_frame=frame,
            store=store,
        )

    def score(self, player_ids: Sequence[int]) -> tuple[list[dict[str, int | float]], int]:
        """Score through the shared core path and return selected global ranks."""

        payload = ranked_prediction_payload(
            self.forecast_frame,
            self.model,
            season_id=self.season_id,
            completion_status=self.completion_status,
        )
        all_predictions = payload["predictions"]
        if not isinstance(all_predictions, list):
            raise ArtifactError("Core prediction payload has an invalid shape")
        selected_ids = set(player_ids)
        selected = [
            prediction
            for prediction in all_predictions
            if isinstance(prediction, dict) and prediction.get("player_id") in selected_ids
        ]
        if len(selected) != len(selected_ids):
            raise HTTPException(status_code=404, detail="One or more players are unavailable")
        return selected, len(all_predictions)


class MetricsRegistry:
    """Thread-safe counters with fixed, low-cardinality label values."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._http: dict[tuple[str, int], int] = {}
        self._predictions_scored = 0

    def observe_http(self, endpoint: str, status_code: int) -> None:
        with self._lock:
            key = (endpoint, status_code)
            self._http[key] = self._http.get(key, 0) + 1

    def observe_predictions(self, count: int) -> None:
        with self._lock:
            self._predictions_scored += count

    def render(self) -> str:
        with self._lock:
            http = sorted(self._http.items())
            scored = self._predictions_scored
        lines = [
            "# HELP fpl_http_requests_total HTTP responses from the bounded service.",
            "# TYPE fpl_http_requests_total counter",
        ]
        for (endpoint, status_code), count in http:
            lines.append(
                f'fpl_http_requests_total{{endpoint="{endpoint}",status="{status_code}"}} {count}'
            )
        lines.extend(
            [
                "# HELP fpl_predictions_scored_total Player predictions returned in memory.",
                "# TYPE fpl_predictions_scored_total counter",
                f"fpl_predictions_scored_total {scored}",
            ]
        )
        return "\n".join(lines) + "\n"


class RequestBodyLimitMiddleware:
    """Reject oversized request bodies before FastAPI parses JSON."""

    def __init__(self, app: Any, max_bytes: int = MAX_REQUEST_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                if int(raw_length) > self.max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send)
                return

        buffered: list[dict[str, Any]] = []
        received = 0
        more_body = True
        while more_body:
            message = await receive()
            if message.get("type") != "http.request":
                buffered.append(message)
                break
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                while message.get("more_body", False):
                    message = await receive()
                await self._reject(send)
                return
            buffered.append(message)
            more_body = bool(message.get("more_body", False))

        index = 0

        async def replay_receive() -> dict[str, Any]:
            nonlocal index
            if index < len(buffered):
                message = buffered[index]
                index += 1
                return message
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _reject(send: Any) -> None:
        body = b'{"detail":"Request body is too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def create_app(
    *,
    artifact_path: str | Path,
    gameweek_dir: str | Path,
    database_path: str | Path,
    season_id: str,
    expected_as_of_gw: int,
    completion_status: str,
    protocol_path: str | Path | None = None,
) -> FastAPI:
    """Create a fail-fast app around one startup-configured deployment."""

    runtime = ServiceRuntime.load(
        artifact_path=artifact_path,
        gameweek_dir=gameweek_dir,
        database_path=database_path,
        season_id=season_id,
        expected_as_of_gw=expected_as_of_gw,
        completion_status=completion_status,
        protocol_path=protocol_path,
    )
    metrics = MetricsRegistry()
    app = FastAPI(
        title="FPL next-gameweek forecast service",
        version="2.0.0",
        debug=False,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(RequestBodyLimitMiddleware)
    app.state.runtime = runtime
    app.state.metrics = metrics

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "Request validation failed"})

    @app.exception_handler(StorageError)
    async def storage_error(_request: Request, _exc: StorageError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "Prediction store unavailable"})

    @app.exception_handler(ForecastingError)
    async def forecasting_error(_request: Request, _exc: ForecastingError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": "Forecast contract rejected the request"},
        )

    @app.exception_handler(Exception)
    async def unhandled_error(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "Internal service error"})

    @app.middleware("http")
    async def observe_requests(request: Request, call_next: Any):
        response = await call_next(request)
        route = request.scope.get("route")
        route_path = getattr(route, "path", None)
        endpoint = (
            route_path
            if route_path
            in {
                "/health",
                "/model",
                "/predict",
                "/predictions/{season_id}/{target_gw}",
                "/metrics",
            }
            else "unmatched"
        )
        metrics.observe_http(endpoint, response.status_code)
        return response

    @app.get("/health")
    def health() -> dict[str, object]:
        database = runtime.store.health()
        return {
            "status": "ok",
            "model": "ready",
            "database": database["status"],
            "schema_version": database["schema_version"],
        }

    @app.get("/model")
    def model() -> dict[str, object]:
        return {
            "artifact_format": ARTIFACT_FORMAT,
            "model_id": runtime.model_registration.model_id,
            "artifact_sha256": runtime.artifact_sha256,
            "season_id": runtime.season_id,
            "feature_schema_version": runtime.model.feature_schema_version,
            "feature_schema_sha256": runtime.model.feature_schema_sha256,
            "protocol_config_sha256": runtime.model.protocol_config_sha256,
            "training_rows": runtime.model.training_rows,
            "trained_through_target_gw": runtime.model.trained_through_target_gw,
        }

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, object]:
        if request.season_id != runtime.season_id or request.as_of_gw != runtime.expected_as_of_gw:
            raise HTTPException(
                status_code=409,
                detail="Request does not match the configured forecast boundary",
            )
        selected, population = runtime.score(request.player_ids)
        metrics.observe_predictions(len(selected))
        return {
            "season_id": runtime.season_id,
            "as_of_gw": runtime.expected_as_of_gw,
            "target_gw": runtime.expected_as_of_gw + 1,
            "ranking_scope": "configured latest-snapshot population",
            "population": population,
            "predictions": selected,
        }

    @app.get("/predictions/{season_id}/{target_gw}")
    def predictions(
        season_id: Annotated[
            str,
            ApiPath(min_length=7, max_length=7, pattern=_SEASON_PATTERN),
        ],
        target_gw: Annotated[int, ApiPath(ge=1, le=60)],
        limit: Annotated[int, Query(ge=1, le=MAX_STORED_PREDICTIONS)] = 100,
    ) -> dict[str, object]:
        season = validate_season_id(season_id)
        batch = runtime.store.latest_predictions(
            model_id=runtime.model_registration.model_id,
            season_id=season,
            target_gw=target_gw,
            limit=limit,
        )
        if batch is None:
            raise HTTPException(status_code=404, detail="Stored predictions not found")
        return {
            "run_id": batch.run_id,
            "model_id": batch.model_id,
            "season_id": batch.season_id,
            "as_of_gw": batch.as_of_gw,
            "target_gw": batch.target_gw,
            "predictions_file_sha256": batch.predictions_file_sha256,
            "manifest_file_sha256": batch.manifest_file_sha256,
            "created_at": batch.created_at.isoformat(),
            "total_predictions": batch.total_predictions,
            "returned_predictions": len(batch.predictions),
            "predictions": [
                {
                    "player_id": item.player_id,
                    "predicted_points": item.prediction,
                    "rank": item.rank,
                }
                for item in batch.predictions
            ],
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics() -> str:
        return metrics.render()

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpl-forecast-service",
        description="Serve one local, startup-configured FPL forecast deployment.",
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--gameweek-dir", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--expected-as-of-gw", required=True, type=int)
    parser.add_argument(
        "--completion-status",
        required=True,
        choices=[COMPLETION_STATUS],
    )
    parser.add_argument("--config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Launch one process; multi-worker DuckDB operation is not supported."""

    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65_535:
        raise SystemExit("--port must be from 1 to 65535")
    app = create_app(
        artifact_path=args.artifact,
        gameweek_dir=args.gameweek_dir,
        database_path=args.database,
        season_id=args.season,
        expected_as_of_gw=args.expected_as_of_gw,
        completion_status=args.completion_status,
        protocol_path=args.config,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        access_log=True,
        server_header=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
