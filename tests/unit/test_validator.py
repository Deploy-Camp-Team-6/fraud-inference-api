import pytest
import yaml  # type: ignore[import-untyped]
from unittest.mock import patch

from app.services.validator import DependencyValidator, DependencyValidationError


@pytest.fixture
def validator():
    """Returns an instance of the DependencyValidator."""
    return DependencyValidator()


@pytest.fixture
def mock_mlflow_artifacts(tmp_path):
    """Mocks mlflow.artifacts.download_artifacts to use a temporary directory."""

    def downloader(artifact_uri):
        # The artifact URI will be something like "model_uri/MLmodel"
        # We just need the filename part
        filename = artifact_uri.split("/")[-1]
        return tmp_path / filename

    return downloader


def create_mock_model_files(tmp_path, conda_deps=None, mlmodel_flavors=None):
    """Helper to create mock MLmodel and conda.yaml files."""
    if conda_deps is None:
        conda_deps = {"pip": ["scikit-learn==1.4.2", "pandas==2.2.1"]}

    conda_content = {
        "name": "mlflow-env",
        "channels": ["conda-forge"],
        "dependencies": ["python=3.11", {"pip": conda_deps["pip"]}],
    }
    with open(tmp_path / "conda.yaml", "w") as f:
        yaml.dump(conda_content, f)

    if mlmodel_flavors is None:
        mlmodel_flavors = {"python_function": {"env": {"conda": "conda.yaml"}}}

    mlmodel_content = {
        "artifact_path": "model",
        "flavors": mlmodel_flavors,
        "signature": {"inputs": [], "outputs": []},
    }
    with open(tmp_path / "MLmodel", "w") as f:
        yaml.dump(mlmodel_content, f)


def test_validator_sklearn_mismatch(validator, tmp_path, mock_mlflow_artifacts):
    """Test that a minor-version mismatch for scikit-learn raises an error."""
    create_mock_model_files(tmp_path)

    mock_versions = {
        "scikit-learn": "1.5.0",  # Mismatch
        "pandas": "2.2.1",
    }

    with (
        patch("mlflow.artifacts.download_artifacts", side_effect=mock_mlflow_artifacts),
        patch(
            "app.services.validator.version",
            side_effect=lambda lib: mock_versions.get(lib, "1.0.0"),
        ),
    ):
        with pytest.raises(DependencyValidationError) as exc_info:
            validator.validate("fake_model_uri")

        assert len(exc_info.value.errors) == 1
        error = exc_info.value.errors[0]
        assert error.library == "scikit-learn"
        assert error.required == "==1.4.2"
        assert error.running == "1.5.0"
        assert error.policy == "compatible"


def test_validator_pandas_compatible(validator, tmp_path, mock_mlflow_artifacts):
    """Test that a compatible minor version for pandas passes."""
    create_mock_model_files(tmp_path)

    mock_versions = {
        "scikit-learn": "1.4.2",
        "pandas": "2.2.5",  # Compatible minor version
    }

    with (
        patch("mlflow.artifacts.download_artifacts", side_effect=mock_mlflow_artifacts),
        patch(
            "app.services.validator.version",
            side_effect=lambda lib: mock_versions.get(lib, "1.0.0"),
        ),
    ):
        try:
            validator.validate("fake_model_uri")
        except DependencyValidationError as e:
            pytest.fail(f"Validation should have passed but failed with: {e.errors}")


def test_validator_gpu_flavor_fails(validator, tmp_path, mock_mlflow_artifacts):
    """Test that a model with a GPU flavor fails validation in a CPU environment."""
    create_mock_model_files(tmp_path, mlmodel_flavors={"gputest": {}})

    with patch(
        "mlflow.artifacts.download_artifacts", side_effect=mock_mlflow_artifacts
    ):
        with pytest.raises(DependencyValidationError) as exc_info:
            validator.validate("fake_model_uri")

        assert len(exc_info.value.errors) == 1
        error = exc_info.value.errors[0]
        assert error.library == "runtime"
        assert error.required == "cpu"
        assert error.running == "gpu"
