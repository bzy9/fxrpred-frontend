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

# Integrated FXR resource collection: cached properties, no RDKit or network.
RESOURCE_PROPERTIES = {
    "qed": ("QED", "QED"),
    "molecular_weight": ("Molecular weight", "Molecular weight (g/mol)"),
    "tpsa": ("TPSA", "TPSA (Å²)"),
    "hba": ("H-bond acceptors", "HBA"),
    "hbd": ("H-bond donors", "HBD"),
    "clogp": ("LogP", "Calculated LogP"),
}
RESOURCE_LABELS = ("FXR_agonist_label", "FXR_antagonist_label")
RESOURCE_EXPORT_COLUMNS = ["Resource ID", "smiles", "Activity annotations", *RESOURCE_LABELS, *RESOURCE_PROPERTIES]


def load_fxr_resources(data_dir: Path) -> pd.DataFrame:
    import hashlib
    import json
    import numpy as np
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name in ("fxr_ligands.csv", "molecular_properties.csv"):
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
            raise BenchmarkDataError(f"Resource file differs from its manifest: {name}.")
    frame = pd.read_csv(root / "fxr_ligands.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    properties = pd.read_csv(root / "molecular_properties.csv", dtype={"Resource ID": str}, keep_default_na=False, encoding="utf-8-sig")
    required = {"Resource ID", "smiles", *RESOURCE_LABELS, "source_ids", "dataset_membership"}
    if not required.issubset(frame) or not {"Resource ID", *RESOURCE_PROPERTIES, "scaffold_smiles"}.issubset(properties):
        raise BenchmarkDataError("The FXR collection is missing required fields.")
    if frame.empty or len(frame) != manifest["unique_molecules"] or frame["Resource ID"].duplicated().any() or frame["smiles"].duplicated().any():
        raise BenchmarkDataError("The FXR collection must contain unique molecular records.")
    for column in RESOURCE_LABELS:
        if not frame[column].isin(["", "0", "1"]).all():
            raise BenchmarkDataError("Resource task labels must be 0, 1 or blank.")
    if not frame[list(RESOURCE_LABELS)].ne("").any(axis=1).all():
        raise BenchmarkDataError("Every FXR resource must have at least one annotated task.")
    if frame["Resource ID"].tolist() != [str(i) for i in range(1, len(frame) + 1)]:
        raise BenchmarkDataError("Resource IDs must follow the stored sequential numbering.")
    if set(frame["Resource ID"]) != set(properties["Resource ID"]):
        raise BenchmarkDataError("Molecular-property records do not cover the collection.")
    frame = frame.merge(properties, on="Resource ID", how="left", validate="one_to_one")
    for column in RESOURCE_PROPERTIES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[list(RESOURCE_PROPERTIES)].to_numpy()).all():
        raise BenchmarkDataError("The collection contains invalid molecular properties.")
    if not frame["qed"].between(0, 1).all() or not frame["molecular_weight"].gt(0).all():
        raise BenchmarkDataError("Invalid QED or molecular weight.")
    for column in ("tpsa", "hba", "hbd"):
        if not frame[column].ge(0).all():
            raise BenchmarkDataError("Invalid polar-surface or hydrogen-bond descriptor.")
    # Keep source evidence separate from the benchmark-aligned display labels.
    for column in RESOURCE_LABELS:
        frame["raw_" + column] = frame[column]
    frame["annotation_policy"] = "curated"
    for override in manifest.get("display_label_overrides", []):
        mask = frame["smiles"].eq(override["smiles"])
        if int(mask.sum()) != 1:
            raise BenchmarkDataError("A display-label override does not identify one molecular record.")
        for column, value in override["labels"].items():
            if column not in RESOURCE_LABELS or value not in ("", "0", "1"):
                raise BenchmarkDataError("Invalid benchmark display-label override.")
            frame.loc[mask, column] = value
        frame.loc[mask, "annotation_policy"] = override["policy"]
    frame["Activity annotations"] = frame.apply(
        lambda r: "; ".join(name for column, name in zip(RESOURCE_LABELS, ("FXR agonist", "FXR antagonist")) if r[column] == "1"), axis=1,
    )
    frame["Activity annotations"] = frame["Activity annotations"].replace("", "Inactive")
    return frame


def filter_fxr_resources(frame: pd.DataFrame, query: str = "") -> pd.DataFrame:
    needle = query.strip()
    if not needle:
        return frame.copy()
    # A valid plain row number selects the exact Resource ID, not SMILES ring digits.
    if needle.isascii() and needle.isdecimal():
        exact = frame["Resource ID"].eq(str(int(needle)))
        if exact.any():
            return frame.loc[exact].copy()
    mask = frame["smiles"].str.contains(needle, case=True, regex=False, na=False)
    for column in ("Resource ID", "source_ids", "BindingDB MonomerID", "BindingDB Reactant_set_id", "Molecule ChEMBL ID"):
        if column in frame:
            mask |= frame[column].str.contains(needle, case=False, regex=False, na=False)
    return frame.loc[mask].copy()


def resource_csv_bytes(frame: pd.DataFrame) -> bytes:
    # Explicit allowlist: never export provenance or future internal fields.
    return frame.loc[:, RESOURCE_EXPORT_COLUMNS].to_csv(index=False).encode("utf-8-sig")


def scaffold_summary(frame: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    counts = frame.loc[frame["scaffold_smiles"].ne(""), "scaffold_smiles"].value_counts()
    table = counts.rename_axis("Scaffold SMILES").reset_index(name="Molecules")
    table = table.sort_values(["Molecules", "Scaffold SMILES"], ascending=[False, True]).head(limit).reset_index(drop=True)
    table.insert(0, "Scaffold", [f"S{i + 1:02}" for i in range(len(table))])
    table["Fraction (%)"] = table["Molecules"] / len(frame) * 100 if len(frame) else 0.0
    return table



def resource_preview(frame: pd.DataFrame, query: str = "", limit: int = 20) -> pd.DataFrame:
    """Default to complete active records; never conceal explicit search matches."""
    if query.strip():
        return frame.head(limit).copy()
    active = frame[list(RESOURCE_LABELS)].eq("1").any(axis=1)
    task_complete = frame[list(RESOURCE_LABELS)].isin(["0", "1"]).all(axis=1)
    # A reviewed benchmark-only display is complete for its accepted direction.
    # Do not invent a negative label merely to keep this record in the preview.
    task_complete |= frame["annotation_policy"].eq("benchmark_preferred")
    source_complete = frame["source_ids"].str.strip().ne("")
    property_complete = frame[list(RESOURCE_PROPERTIES)].notna().all(axis=1)
    return frame.loc[active & task_complete & source_complete & property_complete].head(limit).copy()


def active_resource_groups(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Each profile uses its own positive task label; shared positives can appear twice."""
    return {
        "AGO": frame.loc[frame["FXR_agonist_label"].eq("1")].copy(),
        "ANT": frame.loc[frame["FXR_antagonist_label"].eq("1")].copy(),
    }
