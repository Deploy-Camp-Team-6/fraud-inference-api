import os
import pickle
import sys
import types

mlflow_module = types.ModuleType("mlflow")
tracking_module = types.ModuleType("mlflow.tracking")
setattr(
    tracking_module,
    "MlflowClient",
    lambda tracking_uri=None: types.SimpleNamespace(),
)

artifacts_module = types.ModuleType("mlflow.artifacts")
setattr(
    artifacts_module,
    "download_artifacts",
    lambda artifact_uri=None: None,
)

model_registry_module = types.ModuleType("mlflow.entities.model_registry")
setattr(model_registry_module, "ModelVersion", object)
entities_module = types.ModuleType("mlflow.entities")
setattr(entities_module, "model_registry", model_registry_module)


class RestException(Exception):
    def __init__(self, error_code: str = ""):
        self.error_code = error_code


class MlflowException(Exception):
    pass


exceptions_module = types.ModuleType("mlflow.exceptions")
setattr(exceptions_module, "RestException", RestException)
setattr(exceptions_module, "MlflowException", MlflowException)

pyfunc_module = types.ModuleType("mlflow.pyfunc")


class _FakePyFuncModel:
    def __init__(self, model):
        self._model = model
        self.metadata = types.SimpleNamespace(
            get_signature=lambda: types.SimpleNamespace(
                to_dict=lambda: {"inputs": [{"name": "f0", "type": "long"}]}
            )
        )

    def predict(self, df):
        return self._model.predict(df)

    def unwrap_python_model(self):
        raise MlflowException()


def _load_model(path: str):
    with open(os.path.join(path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    return _FakePyFuncModel(model)


setattr(pyfunc_module, "PyFuncModel", _FakePyFuncModel)
setattr(pyfunc_module, "load_model", _load_model)

xgboost_module = types.ModuleType("mlflow.xgboost")


def _save_model(model, path: str):
    with open(os.path.join(path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


setattr(xgboost_module, "save_model", _save_model)

setattr(mlflow_module, "tracking", tracking_module)
setattr(mlflow_module, "artifacts", artifacts_module)
setattr(mlflow_module, "entities", entities_module)
setattr(mlflow_module, "exceptions", exceptions_module)
setattr(mlflow_module, "pyfunc", pyfunc_module)
setattr(mlflow_module, "xgboost", xgboost_module)

sys.modules["mlflow"] = mlflow_module
sys.modules["mlflow.tracking"] = tracking_module
sys.modules["mlflow.artifacts"] = artifacts_module
sys.modules["mlflow.entities"] = entities_module
sys.modules["mlflow.entities.model_registry"] = model_registry_module
sys.modules["mlflow.exceptions"] = exceptions_module
sys.modules["mlflow.pyfunc"] = pyfunc_module
sys.modules["mlflow.xgboost"] = xgboost_module
