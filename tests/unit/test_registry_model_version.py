from app.models.registry import MlflowModelSelector, ModelInfo, ModelStage


class DummyModelVersion:
    def __init__(self):
        self._name = "test-model"
        self._version = "1"
        self._current_stage = "Staging"
        self._run_id = "run-123"
        self._source = "s3://models/test-model"
        self._last_updated_timestamp = 1716239029


def test_model_info_validation_without_to_dict():
    selector = MlflowModelSelector()
    dummy = DummyModelVersion()
    version_dict = selector._model_version_to_dict(dummy)
    info = ModelInfo.model_validate(version_dict)
    assert info.name == "test-model"
    assert info.version == "1"
    assert info.stage == ModelStage.STAGING
    assert info.run_id == "run-123"
    assert info.source_uri == "s3://models/test-model"
    assert info.last_updated_timestamp == 1716239029
