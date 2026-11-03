"""Ordered prediction batches and recoverable completed results. No UI/network imports."""

import math

TASKS = ("FXR_ago", "PXR_ago", "LXR_ago", "PPAR_ago")


class BatchPredictionError(RuntimeError):
    def __init__(self, message, partial_response, completed_count, total_count):
        super().__init__(message)
        self.partial_response = partial_response
        self.completed_count = completed_count
        self.total_count = total_count


def _inputs(smiles, batch_size):
    if not isinstance(smiles, list) or not smiles or any(not isinstance(s, str) or not s.strip() for s in smiles):
        raise ValueError("Provide a non-empty list of SMILES strings.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Prediction batch size must be a positive integer.")
    return [s.strip() for s in smiles]


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Backend returned an invalid prediction probability.")


def _object(response):
    if not isinstance(response, dict):
        raise ValueError("Backend returned an invalid response format.")
    if response.get("error"):
        raise RuntimeError(str(response["error"]))


def predict_faca_batches(smiles, model_key, threshold, predict, batch_size=16, on_progress=None):
    smiles = _inputs(smiles, batch_size)
    if isinstance(threshold, bool):
        raise ValueError("Threshold must be between zero and one.")
    threshold = float(threshold)
    _probability(threshold)
    rows, indices, skipped, heatmaps = [], [], [], []
    completed = 0
    batches = 0

    def snapshot(complete=False):
        return {"results_df": list(rows), "heatmap_imgs": [item[2] for item in heatmaps],
                "threshold_used": threshold, "threshold_source": "user_selected",
                "model_key": model_key, "input_count": len(smiles), "valid_count": len(rows),
                "completed_input_count": completed, "complete": complete, "batch_count": batches,
                "result_input_indices": list(indices), "skipped_input_indices": list(skipped),
                "image_encoding": "base64_png"}

    if on_progress:
        on_progress(0, len(smiles))
    for start in range(0, len(smiles), batch_size):
        batch = smiles[start:start + batch_size]
        try:
            result = predict("/predict_faca", smiles_list=batch, model_key=model_key, threshold=threshold)
            _object(result)
            used = result.get("threshold_used")
            _probability(used)
            if not math.isclose(used, threshold, rel_tol=0, abs_tol=1e-9):
                raise ValueError("Backend threshold differs from the selected threshold.")
            incoming = result.get("results_df")
            if not isinstance(incoming, list):
                raise ValueError("Backend prediction rows must be a list.")
            if result.get("valid_count", len(incoming)) != len(incoming):
                raise ValueError("Backend valid_count differs from its result count.")
            if result.get("input_count", len(batch)) != len(batch):
                raise ValueError("Backend input_count differs from this batch.")
            # FACA may omit invalid structures. Match an ordered subsequence,
            # retaining duplicate occurrences rather than joining by SMILES.
            positions, cursor = [], 0
            for row in incoming:
                if not isinstance(row, dict):
                    raise ValueError("Invalid FACA result row.")
                _probability(row.get("预测概率"))
                if row.get("预测分类") not in ("活性分子", "非活性分子"):
                    raise ValueError("Invalid FACA prediction label.")
                while cursor < len(batch) and batch[cursor] != row.get("SMILES"):
                    cursor += 1
                if cursor == len(batch):
                    raise ValueError("FACA result order differs from input.")
                positions.append(cursor)
                cursor += 1
            candidates = []
            image_rows = result.get("heatmap_imgs", [])
            if not isinstance(image_rows, list):
                raise ValueError("Invalid heatmap response.")
            active = sorted(
                [(row["预测概率"], start + position, row["SMILES"])
                 for row, position in zip(incoming, positions) if row["预测分类"] == "活性分子"],
                reverse=True,
            )
            for image in image_rows:
                if not isinstance(image, dict) or not image.get("atom_image") or not image.get("frag_image"):
                    raise ValueError("Incomplete attention heatmap.")
                match = next((i for i, entry in enumerate(active) if entry[2] == image.get("smiles")), None)
                if match is None:
                    raise ValueError("Heatmap does not correspond to an active result.")
                probability, index, _ = active.pop(match)
                candidates.append((probability, index, image))
            # Commit only after the whole batch passes validation.
            rows.extend(incoming)
            indices.extend(start + position + 1 for position in positions)
            position_set = set(positions)
            skipped.extend(start + position + 1 for position in range(len(batch)) if position not in position_set)
            heatmaps = sorted(heatmaps + candidates, key=lambda item: (item[0], item[1]), reverse=True)[:3]
            completed += len(batch)
            batches += 1
        except Exception as exc:
            raise BatchPredictionError(str(exc), snapshot(), completed, len(smiles)) from exc
        if on_progress:
            on_progress(completed, len(smiles))
    return snapshot(complete=True)


def predict_ortho_batches(smiles, model_key, predict, batch_size=4, on_progress=None):
    smiles = _inputs(smiles, batch_size)
    rows, completed, batches = [], 0, 0

    def snapshot(complete=False):
        return {"results_list": list(rows), "input_count": len(smiles),
                "completed_input_count": completed, "complete": complete, "batch_count": batches}

    if on_progress:
        on_progress(0, len(smiles))
    for start in range(0, len(smiles), batch_size):
        batch = smiles[start:start + batch_size]
        try:
            result = predict("/predict_ortho", smiles_list=batch, model_key=model_key)
            _object(result)
            incoming = result.get("results_list")
            if not isinstance(incoming, list) or len(incoming) != len(batch):
                raise ValueError("Ortho-MoE result count differs from input.")
            for row, smi in zip(incoming, batch):
                if not isinstance(row, dict) or row.get("SMILES") != smi:
                    raise ValueError("Ortho-MoE result order differs from input.")
                for task in TASKS:
                    _probability(row.get(task + "_prob"))
                    if row.get(task) not in ("Active", "Inactive"):
                        raise ValueError("Missing Ortho-MoE task label.")
            rows.extend(incoming)
            completed += len(batch)
            batches += 1
        except Exception as exc:
            raise BatchPredictionError(str(exc), snapshot(), completed, len(smiles)) from exc
        if on_progress:
            on_progress(completed, len(smiles))
    return snapshot(complete=True)
