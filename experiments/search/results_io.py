import csv
import os
import tempfile
from pathlib import Path


def atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    """
    Atomically write a CSV file.

    Data is written to a temporary file in the same directory,
    flushed to disk, then moved into place with os.replace().
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

            writer = csv.DictWriter(
                temp_file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(rows)

            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(
            temp_path,
            path,
        )

    except Exception:
        if (
            temp_path is not None
            and temp_path.exists()
        ):
            temp_path.unlink()

        raise
