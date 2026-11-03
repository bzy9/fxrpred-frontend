"""Local benchmark data, literal search and source-free CSV exports.

This module has no Streamlit, RDKit, model or network dependency. Molecular
profiles are read from a precomputed table prepared alongside the datasets.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


SOURCE_COLUMN = "来源"
TASK_LABELS = {
    "FXR_ago": "FXR",
    "PXR_ago": "PXR",
    "LXR_ago": "LXR",
    "PPAR_ago": "PPARG",
}
DATASETS = {
    "FXR_AGO": {
        "title": "FXR Agonist",
        "model": "FACA-GIN",
        "description": "A standardized, deduplicated binary benchmark for FXR agonist classification.",
        "labels": ("label",),
        "download_columns": ("smiles", "label"),
        "positive": "Agonist",
        "negative": "Non-agonist",
    },
    "FXR_ANT": {
        "title": "FXR Antagonist",
        "model": "FACA-GIN",
        "description": "A standardized, deduplicated binary benchmark for FXR antagonist classification.",
        "labels": ("label",),
        "download_columns": ("smiles", "label"),
        "positive": "Antagonist",
        "negative": "Non-antagonist",
    },
    "MTL": {
        "title": "Multi-task",
        "model": "Ortho-MoE",
        "description": "A four-task agonist benchmark with the original training, validation and test assignments.",
        "labels": tuple(TASK_LABELS),
        "download_columns": ("smiles", "group", *TASK_LABELS),
        "positive": "Agonist",
        "negative": "Non-agonist",
    },
}
DESCRIPTORS = {
    "molecular_weight": ("Molecular weight", "Molecular weight (g/mol)"),
    "clogp": ("cLogP", "Calculated LogP"),
    "tpsa": ("TPSA", "Topological polar surface area (Å²)"),
}


class BenchmarkDataError(ValueError):
    """A required benchmark column or label is invalid."""


def dataset_spec(dataset_key: str) -> Dict:
    if dataset_key not in DATASETS:
        raise BenchmarkDataError("Unknown benchmark dataset.")
    return DATASETS[dataset_key]


def load_benchmark_data(data_dir: Path, dataset_key: str) -> pd.DataFrame:
    spec = dataset_spec(dataset_key)
    path = Path(data_dir) / f"{dataset_key}.csv"
    if not path.is_file():
        raise BenchmarkDataError(f"The {dataset_key} benchmark file is missing.")
    # Keep the stored label text (including 0.0/1.0) and missing values unchanged.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = set(spec["download_columns"]) | {SOURCE_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise BenchmarkDataError(f"{dataset_key} is missing columns: {', '.join(sorted(missing))}.")
    if frame.empty or frame["smiles"].str.strip().eq("").any():
        raise BenchmarkDataError(f"{dataset_key} contains no data or has empty SMILES.")
    for label in spec["labels"]:
        present = frame[label].str.strip().ne("")
        numeric = pd.to_numeric(frame.loc[present, label], errors="coerce")
        if not numeric.isin([0, 1]).all():
            raise BenchmarkDataError(f"{dataset_key}: {label} must contain only 0, 1 or missing labels.")
        if dataset_key != "MTL" and not present.all():
            raise BenchmarkDataError(f"{dataset_key} contains missing binary labels.")
    return frame


def label_counts(frame: pd.DataFrame, label_column: str) -> Dict[str, int]:
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    return {
        "positive": int(labels.eq(1).sum()),
        "negative": int(labels.eq(0).sum()),
        "missing": int(labels.isna().sum()),
    }


def task_summary(frame: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    rows = []
    for label in dataset_spec(dataset_key)["labels"]:
        counts = label_counts(frame, label)
        labeled = counts["positive"] + counts["negative"]
        rows.append({
            "Task": TASK_LABELS.get(label, DATASETS[dataset_key]["title"]),
            "Positive": counts["positive"],
            "Negative": counts["negative"],
            "Missing": counts["missing"],
            "Labeled": labeled,
            "Coverage": labeled / len(frame) if len(frame) else 0,
        })
    return pd.DataFrame(rows)


def filter_benchmark(
    frame: pd.DataFrame,
    query: str = "",
    search_field: str = "SMILES or source ID",
    label_column: str = "label",
    label_filter: str = "All labels",
    group_filter: str = "All splits",
) -> pd.DataFrame:
    """Literal case-sensitive SMILES search; source IDs ignore case.

    SMILES symbols are not regular expressions. This is not a substructure or
    molecular-equivalence search. No labels, splits or original rows are edited.
    """
    mask = pd.Series(True, index=frame.index)
    needle = query.strip()
    if needle:
        smiles_mask = frame["smiles"].str.contains(needle, case=True, regex=False, na=False)
        source_mask = frame[SOURCE_COLUMN].str.contains(needle, case=False, regex=False, na=False)
        if search_field == "SMILES":
            mask &= smiles_mask
        elif search_field == "Source ID":
            mask &= source_mask
        else:
            mask &= smiles_mask | source_mask
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    if label_filter == "Positive (1)":
        mask &= labels.eq(1)
    elif label_filter == "Negative (0)":
        mask &= labels.eq(0)
    elif label_filter == "Missing label":
        mask &= labels.isna()
    if group_filter != "All splits" and "group" in frame.columns:
        mask &= frame["group"].eq(group_filter)
    return frame.loc[mask].copy()


def preview_frame(frame: pd.DataFrame, dataset_key: str, limit: int = 20) -> pd.DataFrame:
    columns = [*dataset_spec(dataset_key)["download_columns"], SOURCE_COLUMN]
    preview = frame.loc[:, columns].head(limit).copy()
    preview[SOURCE_COLUMN] = preview[SOURCE_COLUMN].replace("", "Not available")
    return preview


def benchmark_csv_bytes(frame: pd.DataFrame, dataset_key: str) -> bytes:
    """An explicit allowlist prevents provenance or future helper-field leaks."""
    columns = list(dataset_spec(dataset_key)["download_columns"])
    return frame.loc[:, columns].to_csv(index=False).encode("utf-8-sig")


def dataset_readme(frame: pd.DataFrame, dataset_key: str) -> str:
    spec = dataset_spec(dataset_key)
    label_definition = (
        "Each task column: 1 = agonist, 0 = non-agonist, blank = unannotated. "
        "Missing labels are not negative observations. The original group column "
        "contains training, val and test assignments."
        if dataset_key == "MTL"
        else f"label: 1 = {spec['positive'].lower()}, 0 = {spec['negative'].lower()}."
    )
    return (
        f"# {spec['title']} benchmark\n\n"
        f"Model: {spec['model']}\n\nRecords: {len(frame):,}\n\n"
        "Sources: NURA and ChEMBL.\n\n "
        "Molecular records were standardized and deduplicated during benchmark "
        "preparation. SMILES are retained exactly as supplied in the final benchmark.\n\n"
        f"{label_definition}\n\n"
        f"CSV columns: {', '.join(spec['download_columns'])}.\n\n"
        "Molecular-property profiles use the largest organic component, chosen by "
        "heavy-atom count and then molecular weight. These descriptive profiles "
        "do not establish model performance or experimental activity.\n"
    )


def load_descriptor_cache(data_dir: Path) -> Optional[pd.DataFrame]:
    path = Path(data_dir) / "descriptor_cache.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path, dtype={"smiles": str}, encoding="utf-8-sig")
    if not {"smiles", *DESCRIPTORS}.issubset(frame.columns) or frame["smiles"].duplicated().any():
        return None
    for column in DESCRIPTORS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["smiles", *DESCRIPTORS]]


def descriptor_profiles(frame: pd.DataFrame, cache: pd.DataFrame, label_column: str) -> pd.DataFrame:
    profiles = frame[["smiles", label_column]].merge(cache, on="smiles", how="left", validate="many_to_one")
    labels = pd.to_numeric(profiles[label_column], errors="coerce")
    profiles["Label"] = "Missing label"
    profiles.loc[labels.eq(1), "Label"] = "Positive (1)"
    profiles.loc[labels.eq(0), "Label"] = "Negative (0)"
    return profiles
