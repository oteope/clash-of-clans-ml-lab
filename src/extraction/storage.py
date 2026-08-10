"""
Storage module for raw JSON extraction data.

Saves raw API responses as immutable source-of-truth files under data/raw/.
"""
import json
import logging
import pathlib

logger = logging.getLogger(__name__)


def save_raw_json(data: dict, category: str, identifier: str) -> None:
    """
    Save raw JSON dictionary to a file under data/raw/{category}/{identifier}.json.

    If data is None (e.g., due to a 404), logs a warning and returns without writing.
    Creates parent directories as needed.
    """
    if data is None:
        logger.warning(
            "No data to save for category=%s identifier=%s. Skipping.", category, identifier
        )
        return

    # Build the target path using pathlib for modern path handling.
    target_dir = pathlib.Path("data") / "raw" / category
    # Ensure the directory exists (creates parents if needed).
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / f"{identifier}.json"

    # Write the raw dictionary without any transformation.
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.debug("Raw data saved to %s", file_path)
