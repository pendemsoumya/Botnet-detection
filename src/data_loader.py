"""Data loading and validation utilities for the botnet dataset."""
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "attack",
    "pkSeqID",
    "category",
    "subcategory",
    "flgs",
    "proto",
    "saddr",
    "sport",
    "daddr",
    "dport",
    "state",
}


def validate_dataset_columns(dataset):
    """Return a sorted list of required columns that are missing."""
    return sorted(REQUIRED_COLUMNS.difference(dataset.columns))


def load_dataset(dataset_path="data/UNSW_2018_IoT_Botnet_Full5pc_4.csv"):
    """
    Load dataset from CSV and validate required columns.

    Args:
        dataset_path: Path to CSV file

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If dataset is not found
        ValueError: If required columns are missing
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info("Loading %s", dataset_path)
    dataset = pd.read_csv(dataset_path, low_memory=False)

    missing_columns = validate_dataset_columns(dataset)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    logger.info(
        "Loaded %s records, %s features", dataset.shape[0], dataset.shape[1]
    )
    return dataset


def inspect_dataset(dataset_path):
    """Inspect dataset headers and row count without loading the full file."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    preview = pd.read_csv(dataset_path, nrows=5, low_memory=False)
    missing_columns = validate_dataset_columns(preview)

    return {
        "path": dataset_path,
        "rows_previewed": len(preview),
        "columns": list(preview.columns),
        "column_count": len(preview.columns),
        "missing_columns": missing_columns,
        "is_valid": not missing_columns,
    }


def get_dataset_info(dataset):
    """Collect and log dataset statistics."""
    info = {
        "rows": int(dataset.shape[0]),
        "columns": int(dataset.shape[1]),
        "missing_values": int(dataset.isnull().sum().sum()),
    }

    if "attack" in dataset.columns:
        labels, counts = np.unique(dataset["attack"].to_numpy().ravel(), return_counts=True)
        distribution = dict(zip(labels.tolist(), counts.tolist()))

        normal_count = int(distribution.get(0, 0))
        attack_count = int(distribution.get(1, 0))
        info["class_distribution"] = {
            "normal": normal_count,
            "attack": attack_count,
        }

        if normal_count:
            info["imbalance_ratio"] = attack_count / normal_count
            logger.info(
                "Class distribution - Normal: %s, Attack: %s",
                normal_count,
                attack_count,
            )
            logger.info(
                "Imbalance ratio: %.2f:1 (Attack:Normal)",
                info["imbalance_ratio"],
            )

    logger.info("Missing values: %s", info["missing_values"])
    return info
