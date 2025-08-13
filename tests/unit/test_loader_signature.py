import types
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from app.models.loader import ModelLoader


class DummySignature:
    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return self._data


class DummyModel:
    def __init__(self, metadata):
        self.metadata = metadata
        self.called_with = None

    def predict(self, df: pd.DataFrame):
        self.called_with = df
        return np.zeros(len(df))


def _mock_model_info():
    return types.SimpleNamespace(
        version="1",
        stage=types.SimpleNamespace(value="Production"),
        run_id="1",
        source_uri="dummy",
    )


def test_load_model_with_get_signature():
    signature_dict = {"inputs": [{"name": "f1", "type": "long"}], "outputs": []}
    signature_obj = DummySignature(signature_dict)
    metadata = types.SimpleNamespace(get_signature=lambda: signature_obj)
    model = DummyModel(metadata)

    loader = ModelLoader()
    model_info = _mock_model_info()

    with (
        patch.object(loader.selector, "select_model_version", return_value=model_info),
        patch.object(loader.validator, "validate"),
        patch("app.models.loader.mlflow.pyfunc.load_model", return_value=model),
        patch("app.models.loader.MODEL_WARMUP_LATENCY_SECONDS", MagicMock()),
        patch("app.models.loader.MODEL_LOAD_SUCCESS_TOTAL", MagicMock()),
    ):
        bundle = loader.load_model_bundle("key", "model")

    assert bundle["signature"] == signature_dict
    assert list(model.called_with.columns) == ["f1"]


def test_load_model_with_signature_attr():
    signature_dict = {"inputs": [{"name": "f2", "type": "float"}], "outputs": []}
    signature_obj = DummySignature(signature_dict)
    metadata = types.SimpleNamespace(signature=signature_obj)
    model = DummyModel(metadata)

    loader = ModelLoader()
    model_info = _mock_model_info()

    with (
        patch.object(loader.selector, "select_model_version", return_value=model_info),
        patch.object(loader.validator, "validate"),
        patch("app.models.loader.mlflow.pyfunc.load_model", return_value=model),
        patch("app.models.loader.MODEL_WARMUP_LATENCY_SECONDS", MagicMock()),
        patch("app.models.loader.MODEL_LOAD_SUCCESS_TOTAL", MagicMock()),
    ):
        bundle = loader.load_model_bundle("key", "model")

    assert bundle["signature"] == signature_dict
    assert list(model.called_with.columns) == ["f2"]
