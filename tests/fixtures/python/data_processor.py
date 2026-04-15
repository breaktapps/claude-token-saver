import json
from pathlib import Path


def load_data(file_path: str) -> dict:
    """Load data from a JSON file."""
    content = Path(file_path).read_text()
    return json.loads(content)


def transform_records(records: list[dict]) -> list[dict]:
    """Transform records by normalizing fields."""
    result = []
    for record in records:
        normalized = normalize_fields(record)
        validated = validate_record(normalized)
        result.append(validated)
    return result


class DataProcessor:
    """Processes data files with validation and transformation."""

    def __init__(self, config: dict):
        self.config = config
        self.errors = []

    def process(self, file_path: str) -> list[dict]:
        """Process a single data file."""
        raw = load_data(file_path)
        records = self.extract_records(raw)
        return transform_records(records)

    def extract_records(self, data: dict) -> list[dict]:
        """Extract records from raw data."""
        key = self.config.get("record_key", "items")
        return data.get(key, [])

    def validate_batch(self, records: list[dict]) -> bool:
        """Validate a batch of records."""
        for record in records:
            if not validate_record(record):
                self.errors.append(record)
                return False
        return True
