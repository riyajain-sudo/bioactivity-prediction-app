"""Offline-safe helpers for validating input and invoking PaDEL per request.

The estimator is intentionally not loaded here: descriptor behavior can be tested
with a mocked subprocess and no model deserialization or provider access.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

MAX_UPLOAD_BYTES = 1_000_000
MAX_MOLECULES = 100
DESCRIPTOR_TIMEOUT_SECONDS = 120
DEFAULT_JAVA_HEAP = "1G"


class PredictionInputError(ValueError):
    """Raised when uploaded data or descriptor output violates the contract."""


def parse_input_text(text: str, max_bytes: int = MAX_UPLOAD_BYTES,
                     max_molecules: int = MAX_MOLECULES) -> pd.DataFrame:
    """Parse the tutorial's two-column whitespace format: SMILES then unique ID."""
    if not text or not text.strip():
        raise PredictionInputError("The uploaded file is empty.")
    if len(text.encode("utf-8")) > max_bytes:
        raise PredictionInputError(f"Upload exceeds the {max_bytes}-byte limit.")
    try:
        rows = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None, dtype=str,
                           engine="python", keep_default_na=False, comment="#")
    except Exception as exc:
        raise PredictionInputError("Could not parse whitespace-delimited SMILES and molecule IDs.") from exc
    if rows.empty or rows.shape[1] != 2:
        raise PredictionInputError("Each non-comment row must contain exactly SMILES and one molecule ID.")
    rows.columns = ["smiles", "molecule_name"]
    if len(rows) > max_molecules:
        raise PredictionInputError(f"Upload exceeds the {max_molecules}-molecule limit.")
    if (rows.smiles.str.strip() == "").any() or (rows.molecule_name.str.strip() == "").any():
        raise PredictionInputError("SMILES and molecule IDs must be non-empty.")
    if rows.molecule_name.duplicated().any():
        raise PredictionInputError("Molecule IDs must be unique.")
    return rows.reset_index(drop=True)


def _java_heap() -> str:
    value = os.getenv("PADEL_MAX_HEAP", DEFAULT_JAVA_HEAP)
    if not re.fullmatch(r"[1-9][0-9]{0,3}[mMgG]", value):
        raise PredictionInputError("PADEL_MAX_HEAP must be a bounded value such as 512M or 1G.")
    number, unit = int(value[:-1]), value[-1].upper()
    megabytes = number * (1024 if unit == "G" else 1)
    if not 256 <= megabytes <= 8192:
        raise PredictionInputError("PADEL_MAX_HEAP must be between 256M and 8G.")
    return value


def calculate_descriptors(rows: pd.DataFrame, app_dir: Path,
                          timeout: int = DESCRIPTOR_TIMEOUT_SECONDS) -> pd.DataFrame:
    """Run PaDEL in an isolated temporary directory and return rows in input order.

    PaDEL receives generated IDs rather than user IDs. Its output must contain
    exactly one row per submitted ID; reorder is tolerated by joining on ID,
    while duplicates, omissions, extras, malformed output, and process errors fail closed.
    """
    if rows.empty or not {"smiles", "molecule_name"}.issubset(rows.columns):
        raise PredictionInputError("No validated molecule rows were supplied.")
    app_dir = Path(app_dir).resolve()
    jar = app_dir / "PaDEL-Descriptor" / "PaDEL-Descriptor.jar"
    descriptor_types = app_dir / "PaDEL-Descriptor" / "PubchemFingerprinter.xml"
    schema_path = app_dir / "descriptor_list.csv"
    for required in (jar, descriptor_types, schema_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required application file is missing: {required.name}")
    feature_names = list(pd.read_csv(schema_path, nrows=0).columns)
    if not feature_names:
        raise PredictionInputError("Descriptor feature schema is empty.")

    ids = [f"request_{i:04d}" for i in range(len(rows))]
    with tempfile.TemporaryDirectory(prefix="bioactivity-padel-") as scratch:
        root = Path(scratch)
        input_dir = root / "inputs"
        input_dir.mkdir()
        input_file = input_dir / "molecules.smi"
        output_file = root / "descriptors.csv"
        with input_file.open("w", encoding="utf-8", newline="\n") as stream:
            for smiles, molecule_id in zip(rows.smiles, ids):
                stream.write(f"{smiles}\t{molecule_id}\n")
        command = [
            "java", f"-Xms256M", f"-Xmx{_java_heap()}", "-Djava.awt.headless=true",
            "-jar", str(jar), "-removesalt", "-standardizenitro", "-fingerprints",
            "-descriptortypes", str(descriptor_types), "-dir", str(input_dir),
            "-file", str(output_file), "-threads", "1",
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise PredictionInputError(f"Descriptor calculation exceeded {timeout} seconds.") from exc
        except subprocess.CalledProcessError as exc:
            raise PredictionInputError("Descriptor calculation failed; no predictions were produced.") from exc
        if not output_file.is_file():
            raise PredictionInputError("Descriptor process completed without producing its output file.")
        try:
            descriptors = pd.read_csv(output_file)
        except Exception as exc:
            raise PredictionInputError("Descriptor output is not a valid CSV file.") from exc
        if "Name" not in descriptors.columns:
            raise PredictionInputError("Descriptor output is missing its molecule ID column.")
        if descriptors["Name"].isna().any() or descriptors["Name"].duplicated().any():
            raise PredictionInputError("Descriptor output contains missing or duplicate molecule IDs.")
        expected, observed = set(ids), set(descriptors["Name"].astype(str))
        if observed != expected or len(descriptors) != len(ids):
            raise PredictionInputError("Descriptor output molecule IDs do not match the submitted input.")
        missing = [name for name in feature_names if name not in descriptors.columns]
        if missing:
            raise PredictionInputError(f"Descriptor output is missing {len(missing)} model features.")
        by_id = descriptors.set_index("Name").loc[ids, feature_names]
        result = by_id.reset_index(drop=True)
        result.insert(0, "molecule_name", rows.molecule_name.to_list())
        return result
