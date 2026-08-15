"""Local single-process persistence for immutable operational forecast records."""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from .contracts import canonical_json_sha256, validate_season_id
from .errors import DataContractError, ForecastingError

SCHEMA_VERSION = 1
MAX_ARTIFACT_BYTES = 10_000_000
MAX_PREDICTIONS_FILE_BYTES = 5_000_000
MAX_MANIFEST_FILE_BYTES = 1_000_000
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WRITER_LOCKS_GUARD = threading.Lock()
_WRITER_LOCKS: dict[Path, threading.Lock] = {}


class StorageError(ForecastingError):
    """Raised when the operational store cannot satisfy its contract."""


class StorageConflictError(StorageError):
    """Raised when immutable or idempotent records conflict."""


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise DataContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _safe_token(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_TOKEN_PATTERN.fullmatch(value) is None:
        raise DataContractError(f"{field} must be a safe non-empty identifier")
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DataContractError(f"{field} must be a positive integer")
    return value


def _writer_lock_for(database_path: Path) -> threading.Lock:
    """Share one non-blocking writer lock across store objects for a local path."""

    with _WRITER_LOCKS_GUARD:
        return _WRITER_LOCKS.setdefault(database_path, threading.Lock())


def read_bounded_bytes(
    path: str | Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read at most ``max_bytes`` from one trusted local configured path."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise DataContractError("max_bytes must be a positive integer")
    try:
        with Path(path).open("rb") as handle:
            payload = handle.read(max_bytes + 1)
    except OSError as exc:
        raise StorageError(f"Configured {label} file is unavailable") from exc
    if len(payload) > max_bytes:
        raise DataContractError(f"Configured {label} file exceeds the supported size")
    return payload


@dataclass(frozen=True)
class ModelRegistration:
    """Non-sensitive identity fields for one immutable model artifact."""

    season_id: str
    artifact_format: str
    artifact_sha256: str
    feature_schema_version: str
    feature_schema_sha256: str
    protocol_config_sha256: str
    trained_through_target_gw: int
    training_rows: int

    def __post_init__(self) -> None:
        validate_season_id(self.season_id)
        _safe_token(self.artifact_format, "artifact_format")
        _sha256(self.artifact_sha256, "artifact_sha256")
        _safe_token(self.feature_schema_version, "feature_schema_version")
        _sha256(self.feature_schema_sha256, "feature_schema_sha256")
        _sha256(self.protocol_config_sha256, "protocol_config_sha256")
        _positive_integer(self.trained_through_target_gw, "trained_through_target_gw")
        _positive_integer(self.training_rows, "training_rows")

    @property
    def model_id(self) -> str:
        """Return a deterministic identifier without exposing artifact contents."""

        return "model-" + canonical_json_sha256(self.identity_payload())[:32]

    def identity_payload(self) -> dict[str, object]:
        return {
            "contract": "fpl-model-registration-v1",
            "season_id": self.season_id,
            "artifact_format": self.artifact_format,
            "artifact_sha256": self.artifact_sha256,
            "feature_schema_version": self.feature_schema_version,
            "feature_schema_sha256": self.feature_schema_sha256,
            "protocol_config_sha256": self.protocol_config_sha256,
            "trained_through_target_gw": self.trained_through_target_gw,
            "training_rows": self.training_rows,
        }


@dataclass(frozen=True)
class BatchRequest:
    """Every input component that determines one batch forecast request."""

    model_id: str
    season_id: str
    as_of_gw: int
    target_gw: int
    source_snapshot_sha256: str
    forecast_frame_sha256: str
    feature_schema_sha256: str
    protocol_config_sha256: str
    predictions_file_sha256: str
    manifest_file_sha256: str

    def __post_init__(self) -> None:
        _safe_token(self.model_id, "model_id")
        validate_season_id(self.season_id)
        as_of_gw = _positive_integer(self.as_of_gw, "as_of_gw")
        target_gw = _positive_integer(self.target_gw, "target_gw")
        if target_gw != as_of_gw + 1:
            raise DataContractError("target_gw must be exactly one after as_of_gw")
        for field in (
            "source_snapshot_sha256",
            "forecast_frame_sha256",
            "feature_schema_sha256",
            "protocol_config_sha256",
            "predictions_file_sha256",
            "manifest_file_sha256",
        ):
            _sha256(getattr(self, field), field)

    @property
    def idempotency_key(self) -> str:
        """Hash request inputs only, so changed output under one request conflicts."""

        return canonical_json_sha256(self.identity_payload())

    @property
    def run_id(self) -> str:
        return "run-" + self.idempotency_key[:32]

    def identity_payload(self) -> dict[str, object]:
        """Return causal inputs; exact output-file hashes are replay evidence."""

        return {
            "contract": "fpl-batch-request-v1",
            "model_id": self.model_id,
            "season_id": self.season_id,
            "as_of_gw": self.as_of_gw,
            "target_gw": self.target_gw,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "forecast_frame_sha256": self.forecast_frame_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "protocol_config_sha256": self.protocol_config_sha256,
        }


@dataclass(frozen=True)
class StoredPrediction:
    """One immutable player prediction stored for a batch run."""

    player_id: int
    prediction: float
    rank: int

    def __post_init__(self) -> None:
        _positive_integer(self.player_id, "player_id")
        _positive_integer(self.rank, "rank")
        if isinstance(self.prediction, bool) or not isinstance(self.prediction, (int, float)):
            raise DataContractError("prediction must be numeric")
        if not math.isfinite(float(self.prediction)):
            raise DataContractError("prediction must be finite")

    def payload(self) -> dict[str, int | float]:
        return {
            "player_id": self.player_id,
            "prediction": float(self.prediction),
            "rank": self.rank,
        }


@dataclass(frozen=True)
class BatchWriteResult:
    """Identity and replay state returned after a transactional batch write."""

    run_id: str
    idempotency_key: str
    prediction_count: int
    replayed: bool


@dataclass(frozen=True)
class StoredBatch:
    """Bounded read model for the latest stored target-gameweek batch."""

    run_id: str
    model_id: str
    season_id: str
    as_of_gw: int
    target_gw: int
    predictions_file_sha256: str
    manifest_file_sha256: str
    created_at: datetime
    total_predictions: int
    predictions: tuple[StoredPrediction, ...]


_MIGRATION_V1 = """
CREATE TABLE model_registry (
    model_id VARCHAR PRIMARY KEY,
    season_id VARCHAR NOT NULL,
    artifact_format VARCHAR NOT NULL,
    artifact_sha256 VARCHAR NOT NULL,
    feature_schema_version VARCHAR NOT NULL,
    feature_schema_sha256 VARCHAR NOT NULL,
    protocol_config_sha256 VARCHAR NOT NULL,
    trained_through_target_gw INTEGER NOT NULL CHECK (trained_through_target_gw > 0),
    training_rows BIGINT NOT NULL CHECK (training_rows > 0),
    registered_at TIMESTAMPTZ NOT NULL,
    UNIQUE (season_id, artifact_sha256, feature_schema_sha256, protocol_config_sha256)
);

CREATE TABLE batch_runs (
    run_id VARCHAR PRIMARY KEY,
    idempotency_key VARCHAR NOT NULL UNIQUE,
    model_id VARCHAR NOT NULL REFERENCES model_registry(model_id),
    season_id VARCHAR NOT NULL,
    as_of_gw INTEGER NOT NULL CHECK (as_of_gw > 0),
    target_gw INTEGER NOT NULL CHECK (target_gw = as_of_gw + 1),
    source_snapshot_sha256 VARCHAR NOT NULL,
    forecast_frame_sha256 VARCHAR NOT NULL,
    feature_schema_sha256 VARCHAR NOT NULL,
    protocol_config_sha256 VARCHAR NOT NULL,
    stored_predictions_sha256 VARCHAR NOT NULL,
    predictions_file_sha256 VARCHAR NOT NULL,
    manifest_file_sha256 VARCHAR NOT NULL,
    prediction_count INTEGER NOT NULL CHECK (prediction_count > 0),
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE predictions (
    run_id VARCHAR NOT NULL REFERENCES batch_runs(run_id),
    player_id INTEGER NOT NULL CHECK (player_id > 0),
    prediction DOUBLE NOT NULL,
    rank INTEGER NOT NULL CHECK (rank > 0),
    PRIMARY KEY (run_id, player_id),
    UNIQUE (run_id, rank)
);
"""


class OperationalStore:
    """Configured DuckDB store with one explicit writer inside one local process."""

    def __init__(self, database_path: str | Path, *, read_only: bool = False) -> None:
        self._path = Path(database_path).resolve()
        self._read_only = read_only
        if read_only:
            if not self._path.is_file():
                raise StorageError("Configured operational database is unavailable")
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer_lock = _writer_lock_for(self._path)
        if read_only:
            self.health()
        else:
            self._writer_lock.acquire()
            try:
                self._migrate()
            finally:
                self._writer_lock.release()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self._path), read_only=self._read_only)

    def _migrate(self) -> None:
        connection: duckdb.DuckDBPyConnection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            applied = {
                int(row[0])
                for row in connection.execute("SELECT version FROM schema_migrations").fetchall()
            }
            if any(version > SCHEMA_VERSION for version in applied):
                raise StorageError("Database schema is newer than this application")
            if 1 not in applied:
                connection.execute(_MIGRATION_V1)
                connection.execute(
                    "INSERT INTO schema_migrations VALUES (?, ?)",
                    [1, datetime.now(UTC)],
                )
            connection.execute("COMMIT")
        except duckdb.Error as exc:
            if connection is not None:
                self._rollback_quietly(connection)
            raise StorageError("Operational database migration failed") from exc
        except Exception:
            if connection is not None:
                self._rollback_quietly(connection)
            raise
        finally:
            if connection is not None:
                self._close_quietly(connection)

    @staticmethod
    def _rollback_quietly(connection: duckdb.DuckDBPyConnection) -> None:
        try:
            connection.execute("ROLLBACK")
        except duckdb.Error:
            pass

    @staticmethod
    def _close_quietly(connection: duckdb.DuckDBPyConnection) -> None:
        try:
            connection.close()
        except duckdb.Error:
            pass

    @contextmanager
    def _reader(self) -> Iterator[duckdb.DuckDBPyConnection]:
        self._writer_lock.acquire()
        connection: duckdb.DuckDBPyConnection | None = None
        try:
            connection = self._connect()
            yield connection
        except duckdb.Error as exc:
            raise StorageError("Operational database read failed") from exc
        finally:
            if connection is not None:
                self._close_quietly(connection)
            self._writer_lock.release()

    @contextmanager
    def writer(self) -> Iterator[StoreWriter]:
        """Open the sole local writer transaction and roll back every exception."""

        if self._read_only:
            raise StorageError("This operational store is configured for read-only access")
        if not self._writer_lock.acquire(blocking=False):
            raise StorageConflictError("Another writer transaction is already active")
        connection: duckdb.DuckDBPyConnection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN TRANSACTION")
            yield StoreWriter(connection)
            connection.execute("COMMIT")
        except duckdb.Error as exc:
            if connection is not None:
                self._rollback_quietly(connection)
            raise StorageError("Operational database write failed") from exc
        except Exception:
            if connection is not None:
                self._rollback_quietly(connection)
            raise
        finally:
            if connection is not None:
                self._close_quietly(connection)
            self._writer_lock.release()

    def health(self) -> dict[str, int | str]:
        """Return non-sensitive schema readiness information."""

        with self._reader() as connection:
            version = connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
            connection.execute("SELECT 1 FROM model_registry LIMIT 0")
            connection.execute("SELECT 1 FROM batch_runs LIMIT 0")
            connection.execute("SELECT 1 FROM predictions LIMIT 0")
        if version != SCHEMA_VERSION:
            raise StorageError("Database migration state is incomplete")
        return {"status": "ready", "schema_version": int(version)}

    def counts(self) -> dict[str, int]:
        """Return aggregate row counts without exposing stored identities."""

        with self._reader() as connection:
            models = connection.execute("SELECT count(*) FROM model_registry").fetchone()[0]
            runs = connection.execute("SELECT count(*) FROM batch_runs").fetchone()[0]
            predictions = connection.execute("SELECT count(*) FROM predictions").fetchone()[0]
        return {"models": int(models), "runs": int(runs), "predictions": int(predictions)}

    def require_registered_model(self, model: ModelRegistration) -> str:
        """Require the exact configured artifact registration before serving."""

        with self._reader() as connection:
            existing = connection.execute(
                """
                SELECT season_id, artifact_format, artifact_sha256, feature_schema_version,
                       feature_schema_sha256, protocol_config_sha256,
                       trained_through_target_gw, training_rows
                FROM model_registry WHERE model_id = ?
                """,
                [model.model_id],
            ).fetchone()
        expected = (
            model.season_id,
            model.artifact_format,
            model.artifact_sha256,
            model.feature_schema_version,
            model.feature_schema_sha256,
            model.protocol_config_sha256,
            model.trained_through_target_gw,
            model.training_rows,
        )
        if existing is None:
            raise StorageError("Configured model is not registered in the operational database")
        if tuple(existing) != expected:
            raise StorageConflictError("Configured model registration is inconsistent")
        return model.model_id

    def latest_predictions(
        self,
        *,
        model_id: str,
        season_id: str,
        target_gw: int,
        limit: int,
    ) -> StoredBatch | None:
        """Read a bounded page from the latest immutable run for one target."""

        model = _safe_token(model_id, "model_id")
        season = validate_season_id(season_id)
        target = _positive_integer(target_gw, "target_gw")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise DataContractError("limit must be an integer from 1 to 500")
        with self._reader() as connection:
            run = connection.execute(
                """
                SELECT run_id, model_id, season_id, as_of_gw, target_gw,
                       predictions_file_sha256, manifest_file_sha256,
                       created_at, prediction_count
                FROM batch_runs
                WHERE model_id = ? AND season_id = ? AND target_gw = ?
                ORDER BY created_at DESC, run_id DESC
                LIMIT 1
                """,
                [model, season, target],
            ).fetchone()
            if run is None:
                return None
            rows = connection.execute(
                """
                SELECT player_id, prediction, rank
                FROM predictions
                WHERE run_id = ?
                ORDER BY rank, player_id
                LIMIT ?
                """,
                [run[0], limit],
            ).fetchall()
        return StoredBatch(
            run_id=str(run[0]),
            model_id=str(run[1]),
            season_id=str(run[2]),
            as_of_gw=int(run[3]),
            target_gw=int(run[4]),
            predictions_file_sha256=str(run[5]),
            manifest_file_sha256=str(run[6]),
            created_at=run[7],
            total_predictions=int(run[8]),
            predictions=tuple(
                StoredPrediction(player_id=int(row[0]), prediction=float(row[1]), rank=int(row[2]))
                for row in rows
            ),
        )


class StoreWriter:
    """Mutation surface available only inside ``OperationalStore.writer``."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def register_model(self, model: ModelRegistration) -> str:
        existing = self._connection.execute(
            """
            SELECT season_id, artifact_format, artifact_sha256, feature_schema_version,
                   feature_schema_sha256, protocol_config_sha256,
                   trained_through_target_gw, training_rows
            FROM model_registry WHERE model_id = ?
            """,
            [model.model_id],
        ).fetchone()
        expected = (
            model.season_id,
            model.artifact_format,
            model.artifact_sha256,
            model.feature_schema_version,
            model.feature_schema_sha256,
            model.protocol_config_sha256,
            model.trained_through_target_gw,
            model.training_rows,
        )
        if existing is not None:
            if tuple(existing) != expected:
                raise StorageConflictError("Model identity conflicts with an immutable record")
            return model.model_id
        self._connection.execute(
            """
            INSERT INTO model_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                model.model_id,
                *expected,
                datetime.now(UTC),
            ],
        )
        return model.model_id

    def persist_batch(
        self,
        request: BatchRequest,
        predictions: Sequence[StoredPrediction],
    ) -> BatchWriteResult:
        """Insert one complete immutable batch or replay its identical prior result."""

        ordered = tuple(sorted(predictions, key=lambda item: (item.rank, item.player_id)))
        if not ordered:
            raise DataContractError("batch predictions cannot be empty")
        if len({item.player_id for item in ordered}) != len(ordered):
            raise DataContractError("batch player_id values must be unique")
        if {item.rank for item in ordered} != set(range(1, len(ordered) + 1)):
            raise DataContractError("batch ranks must be unique and consecutive from one")
        expected_ranking = tuple(
            sorted(ordered, key=lambda item: (-float(item.prediction), item.player_id))
        )
        if ordered != expected_ranking:
            raise DataContractError(
                "batch ranks must order prediction descending, then player_id ascending"
            )

        model = self._connection.execute(
            """
            SELECT season_id, feature_schema_sha256, protocol_config_sha256
            FROM model_registry WHERE model_id = ?
            """,
            [request.model_id],
        ).fetchone()
        if model is None:
            raise StorageError("Batch model is not registered")
        if tuple(model) != (
            request.season_id,
            request.feature_schema_sha256,
            request.protocol_config_sha256,
        ):
            raise StorageConflictError("Batch request does not match its registered model")

        stored_predictions_sha256 = canonical_json_sha256(
            {
                "contract": "fpl-stored-predictions-v1",
                "predictions": [item.payload() for item in ordered],
            }
        )
        existing = self._connection.execute(
            """
            SELECT run_id, model_id, season_id, as_of_gw, target_gw,
                   source_snapshot_sha256, forecast_frame_sha256,
                   feature_schema_sha256, protocol_config_sha256,
                   stored_predictions_sha256, predictions_file_sha256,
                   manifest_file_sha256, prediction_count
            FROM batch_runs WHERE idempotency_key = ?
            """,
            [request.idempotency_key],
        ).fetchone()
        expected = (
            request.run_id,
            request.model_id,
            request.season_id,
            request.as_of_gw,
            request.target_gw,
            request.source_snapshot_sha256,
            request.forecast_frame_sha256,
            request.feature_schema_sha256,
            request.protocol_config_sha256,
            stored_predictions_sha256,
            request.predictions_file_sha256,
            request.manifest_file_sha256,
            len(ordered),
        )
        if existing is not None:
            if tuple(existing) != expected:
                raise StorageConflictError(
                    "Idempotency key conflicts with a different batch result"
                )
            return BatchWriteResult(
                run_id=request.run_id,
                idempotency_key=request.idempotency_key,
                prediction_count=len(ordered),
                replayed=True,
            )

        self._connection.execute(
            """
            INSERT INTO batch_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                request.run_id,
                request.idempotency_key,
                request.model_id,
                request.season_id,
                request.as_of_gw,
                request.target_gw,
                request.source_snapshot_sha256,
                request.forecast_frame_sha256,
                request.feature_schema_sha256,
                request.protocol_config_sha256,
                stored_predictions_sha256,
                request.predictions_file_sha256,
                request.manifest_file_sha256,
                len(ordered),
                datetime.now(UTC),
            ],
        )
        for prediction in ordered:
            self._insert_prediction(request.run_id, prediction)
        return BatchWriteResult(
            run_id=request.run_id,
            idempotency_key=request.idempotency_key,
            prediction_count=len(ordered),
            replayed=False,
        )

    def _insert_prediction(self, run_id: str, prediction: StoredPrediction) -> None:
        self._connection.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?)",
            [run_id, prediction.player_id, float(prediction.prediction), prediction.rank],
        )
