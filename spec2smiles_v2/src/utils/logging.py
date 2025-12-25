"""Live training logging utilities.

Provides real-time logging of per-epoch metrics to CSV files during training.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class TrainingLogger:
    """Live logging of per-epoch metrics to CSV file.

    Writes metrics to file immediately after each epoch, allowing
    real-time monitoring of training progress.

    Example:
        logger = TrainingLogger(
            log_dir=Path("logs"),
            model_name="transformer",
            metrics=["train_loss", "val_loss", "train_r2", "val_r2", "lr"]
        )

        for epoch in range(100):
            # ... training code ...
            logger.log_epoch(epoch, {
                "train_loss": 0.234,
                "val_loss": 0.256,
                "train_r2": 0.654,
                "val_r2": 0.632,
                "lr": 0.0001
            })

        logger.close()
    """

    def __init__(
        self,
        log_dir: Path,
        model_name: str,
        metrics: List[str],
        timestamp: Optional[str] = None,
    ):
        """Initialize training logger.

        Args:
            log_dir: Directory for log files
            model_name: Name of the model (e.g., "transformer", "vae")
            metrics: List of metric names to log
            timestamp: Optional timestamp string (defaults to current time)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.metrics = metrics

        # Generate timestamp for file name
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp

        # Create log file path
        self.log_path = self.log_dir / f"train_{model_name}_{timestamp}_epochs.csv"

        # Open file and write header
        self._file = open(self.log_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "epoch"] + metrics)
        self._file.flush()

        self._closed = False

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for a single epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            metrics: Dictionary mapping metric names to values
        """
        if self._closed:
            raise RuntimeError("Logger has been closed")

        timestamp = datetime.now().isoformat()
        row = [timestamp, epoch]

        for metric_name in self.metrics:
            value = metrics.get(metric_name, float("nan"))
            row.append(f"{value:.6f}" if isinstance(value, float) else str(value))

        self._writer.writerow(row)
        self._file.flush()  # Ensure immediate write to disk

    def close(self) -> None:
        """Close the log file."""
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def filepath(self) -> Path:
        """Return the path to the log file."""
        return self.log_path


def get_epoch_log_path(log_dir: Path, model_name: str, timestamp: str) -> Path:
    """Get the expected path for an epoch log file.

    Args:
        log_dir: Directory for log files
        model_name: Name of the model
        timestamp: Timestamp string

    Returns:
        Path to the epoch log file
    """
    return log_dir / f"train_{model_name}_{timestamp}_epochs.csv"
