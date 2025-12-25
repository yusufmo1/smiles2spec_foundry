"""Data loading and validation service."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import settings
from src.utils.exceptions import DataError


class DataLoaderService:
    """Load and validate spectral data from JSONL files."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize data loader.

        Args:
            data_dir: Path to data directory (defaults to settings.input_path)
        """
        self.data_dir = Path(data_dir) if data_dir else settings.input_path

    def load_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load JSONL file.

        Args:
            filepath: Path to JSONL file

        Returns:
            List of parsed JSON objects

        Raises:
            DataError: If file cannot be read or parsed
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise DataError(f"File not found: {filepath}")

        data = []
        with open(filepath, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise DataError(f"JSON parsing error at line {line_num}: {e}")
        return data

    def save_jsonl(self, data: List[Dict[str, Any]], filepath: Path) -> None:
        """Save data to JSONL file.

        Args:
            data: List of dictionaries to save
            filepath: Output path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    @staticmethod
    def validate_entry(entry: Dict[str, Any], min_peaks: int = 3) -> bool:
        """Validate a data entry.

        Args:
            entry: Dictionary with sample data
            min_peaks: Minimum number of peaks required

        Returns:
            True if entry is valid
        """
        if "smiles" not in entry or not entry["smiles"]:
            return False
        if "peaks" not in entry or not entry["peaks"]:
            return False
        if len(entry["peaks"]) < min_peaks:
            return False
        return True

    def load_raw_data(
        self, filepath: Path | None = None, min_peaks: int = 3
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Load and validate raw spectral data.

        Args:
            filepath: Path to JSONL file (defaults to spectral_data.jsonl)
            min_peaks: Minimum peaks for valid spectrum

        Returns:
            Tuple of (valid data, total entries loaded)
        """
        if filepath is None:
            filepath = self.data_dir / "spectral_data.jsonl"

        raw_data = self.load_jsonl(filepath)
        valid_data = [
            entry for entry in raw_data if self.validate_entry(entry, min_peaks)
        ]
        return valid_data, len(raw_data)

    def load_processed_splits(
        self, data_dir: Path | None = None
    ) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
        """Load preprocessed train/val/test splits.

        Args:
            data_dir: Directory containing processed data

        Returns:
            Tuple of (train_data, val_data, test_data, metadata)
        """
        data_dir = Path(data_dir) if data_dir else self.data_dir

        train_data = self.load_jsonl(data_dir / "train_data.jsonl")
        val_data = self.load_jsonl(data_dir / "val_data.jsonl")
        test_data = self.load_jsonl(data_dir / "test_data.jsonl")

        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        return train_data, val_data, test_data, metadata

    @staticmethod
    def extract_features_and_targets(
        data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract spectral features and descriptor targets from preprocessed data.

        Args:
            data: List of preprocessed sample dictionaries

        Returns:
            Tuple of (spectra, descriptors, smiles_list)
        """
        spectra = np.array(
            [sample["spectrum"] for sample in data], dtype=np.float32
        )
        descriptors = np.array(
            [sample["descriptors"] for sample in data], dtype=np.float32
        )
        smiles_list = [sample["smiles"] for sample in data]

        return spectra, descriptors, smiles_list
