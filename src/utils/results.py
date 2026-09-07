"""Utilities for saving experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json_results(
    results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save experiment results as a formatted JSON file.

    Parameters
    ----------
    results:
        Dictionary containing experiment results.

    output_path:
        Destination JSON file.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
        )