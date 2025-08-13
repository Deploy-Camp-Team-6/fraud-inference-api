import importlib.util
import sys
import types

if importlib.util.find_spec("mlflow") is None:
    mlflow_module = types.ModuleType("mlflow")
    tracking_module = types.ModuleType("mlflow.tracking")
    tracking_module.MlflowClient = lambda tracking_uri=None: types.SimpleNamespace()

    artifacts_module = types.ModuleType("mlflow.artifacts")
    artifacts_module.download_artifacts = lambda artifact_uri=None: None

    model_registry_module = types.ModuleType("mlflow.entities.model_registry")
    model_registry_module.ModelVersion = object
    entities_module = types.ModuleType("mlflow.entities")
    entities_module.model_registry = model_registry_module

    class RestException(Exception):
        def __init__(self, error_code: str = ""):
            self.error_code = error_code

    exceptions_module = types.ModuleType("mlflow.exceptions")
    exceptions_module.RestException = RestException

    mlflow_module.tracking = tracking_module
    mlflow_module.artifacts = artifacts_module
    mlflow_module.entities = entities_module
    mlflow_module.exceptions = exceptions_module

    sys.modules["mlflow"] = mlflow_module
    sys.modules["mlflow.tracking"] = tracking_module
    sys.modules["mlflow.artifacts"] = artifacts_module
    sys.modules["mlflow.entities"] = entities_module
    sys.modules["mlflow.entities.model_registry"] = model_registry_module
    sys.modules["mlflow.exceptions"] = exceptions_module
