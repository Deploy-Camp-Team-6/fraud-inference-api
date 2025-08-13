import tempfile
import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
import pytest

from app.models.loader import _pick_predict_fn
from mlflow.exceptions import MlflowException


def test_xgboost_model_loads_without_exception():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = xgb.XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, eval_metric="logloss")
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.xgboost.save_model(model, tmpdir)
        loaded = mlflow.pyfunc.load_model(tmpdir)

        with pytest.raises(MlflowException):
            loaded.unwrap_python_model()

        predict_fn = _pick_predict_fn(loaded)
        df = pd.DataFrame(X)
        preds = predict_fn(df)
        assert len(preds) == len(X)
