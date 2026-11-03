"""Streamlit-to-Gradio transport. No connection is made during module import."""

import ast
import inspect
import json
import math
import os

from gradio_client import Client
import streamlit as st
from prediction_batches import BatchPredictionError, predict_faca_batches, predict_ortho_batches

DEFAULT_SPACE_ID = "gzLiang/fxr-model-service"


def _setting(name, default=None):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        return default


@st.cache_resource(show_spinner=False, ttl=600)
def _create_client(space_id, token):
    kwargs = {"verbose": False}
    if token:
        # The configured client version is pinned; support the legacy spelling
        # as well for users retaining an existing local frontend environment.
        parameters = inspect.signature(Client).parameters
        name = "token" if "token" in parameters else "hf_token"
        kwargs[name] = token
    return Client(space_id, **kwargs)


def _predict(api_name, **inputs):
    space_id = str(_setting("HF_SPACE_ID", DEFAULT_SPACE_ID)).strip()
    if not space_id:
        raise RuntimeError("HF_SPACE_ID is not configured.")
    client = _create_client(space_id, _setting("HF_TOKEN") or None)
    response = client.predict(api_name=api_name, **inputs)
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            response = ast.literal_eval(response)
    if not isinstance(response, dict):
        raise RuntimeError("Backend returned an invalid response format.")
    if response.get("error"):
        raise RuntimeError(str(response["error"]))
    return response


def get_faca_recommended_threshold(model_key):
    try:
        return _predict("/get_faca_threshold", model_key=model_key.upper())
    except Exception as exc:
        raise RuntimeError(f"读取 FACA-GIN 推荐阈值失败: {exc}") from exc


def _batch_size(name, default):
    value = _setting(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    result = int(value)
    if result < 1 or str(value).strip() != str(result):
        raise ValueError(f"{name} must be a positive integer.")
    return result


def call_faca_gin_api(smiles_list, model_key, threshold, on_progress=None):
    try:
        return predict_faca_batches(
            smiles_list, model_key.upper(), threshold, _predict,
            batch_size=_batch_size("FACA_API_BATCH_SIZE", 16), on_progress=on_progress,
        )
    except BatchPredictionError:
        raise
    except Exception as exc:
        raise RuntimeError(f"网络连接或 FACA-GIN 推理失败: {exc}") from exc


def call_ortho_moe_api(smiles_list, model_key="ORTHO", on_progress=None):
    try:
        return predict_ortho_batches(
            smiles_list, model_key.upper(), _predict,
            batch_size=_batch_size("ORTHO_API_BATCH_SIZE", 4), on_progress=on_progress,
        )
    except BatchPredictionError:
        raise
    except Exception as exc:
        raise RuntimeError(f"网络连接或 Ortho-MoE 推理失败: {exc}") from exc
