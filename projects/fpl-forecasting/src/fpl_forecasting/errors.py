"""Project-specific validation errors."""


class ForecastingError(Exception):
    """Base error for the forecasting package."""


class DataContractError(ForecastingError):
    """Raised when an input violates the gameweek data contract."""


class EmptyGameweekError(DataContractError):
    """Raised when a gameweek has no player records."""


class DuplicateGameweekError(DataContractError):
    """Raised when more than one file claims the same gameweek."""


class NonConsecutiveGameweeksError(DataContractError):
    """Raised when the supplied gameweek sequence has a gap."""


class LeakageError(ForecastingError):
    """Raised when an evaluation frame violates the as-of boundary."""


class SplitError(ForecastingError):
    """Raised when a chronological split cannot be formed safely."""


class ArtifactError(ForecastingError):
    """Raised when a saved model artifact is malformed or incompatible."""
