from importlib.metadata import version, PackageNotFoundError

import mlflow
import yaml  # type: ignore[import-untyped]
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from typing import Literal

from pydantic import BaseModel

from app.core.logging import get_logger

logger = get_logger(__name__)

# --- Configuration ---

# Rules: library -> "exact" or "compatible"
VALIDATION_RULES: dict[str, Literal["exact", "compatible"]] = {
    "scikit-learn": "exact",
    "xgboost": "exact",
    "lightgbm": "exact",
    "pandas": "compatible",
    "numpy": "compatible",
    "mlflow": "compatible",
}

# --- Schemas & Exceptions ---


class DependencyError(BaseModel):
    """A single dependency validation error."""

    library: str
    required: str
    running: str
    policy: Literal["exact", "compatible"]
    action: Literal["fail"] = "fail"


class DependencyValidationError(Exception):
    """Custom exception for dependency validation errors."""

    def __init__(self, message: str, errors: list[DependencyError]):
        super().__init__(message)
        self.errors = errors


# --- Validator ---


class DependencyValidator:
    """
    Validates the dependencies of an MLflow model against the current environment.
    """

    def validate(self, model_uri: str) -> None:
        """
        Parses model dependencies and compares them against the current environment.
        Raises DependencyValidationError if there are mismatches.
        """
        logger.info("Starting dependency validation", model_uri=model_uri)
        errors = []

        try:
            model_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"{model_uri}/MLmodel"
            )
            with open(model_path) as f:
                mlmodel = yaml.safe_load(f)
        except Exception as e:
            logger.error(
                "Failed to download or parse MLmodel file",
                model_uri=model_uri,
                exc_info=e,
            )
            raise DependencyValidationError(
                f"Failed to download or parse MLmodel file from {model_uri}: {e}",
                [],
            ) from e

        # Check for GPU requirements in a CPU-only environment
        self._check_for_gpu_flavors(mlmodel)

        # Find conda.yaml or requirements.txt
        dependencies = self._get_dependencies_from_mlmodel(mlmodel, model_uri)
        if not dependencies:
            logger.warning(
                "No supported dependency file found for model", model_uri=model_uri
            )
            return

        for lib, rule in VALIDATION_RULES.items():
            required_spec = self._find_required_spec(lib, dependencies)
            if not required_spec:
                logger.debug("Library not found in model dependencies", library=lib)
                continue

            try:
                installed_version = Version(version(lib))
            except PackageNotFoundError:
                errors.append(
                    DependencyError(
                        library=lib,
                        required=required_spec,
                        running="Not installed",
                        policy=rule,
                    )
                )
                continue

            if not self._is_version_valid(required_spec, installed_version, rule):
                errors.append(
                    DependencyError(
                        library=lib,
                        required=required_spec,
                        running=str(installed_version),
                        policy=rule,
                    )
                )

        if errors:
            error_details = [err.model_dump_json() for err in errors]
            logger.error(
                "Dependency validation failed",
                model_uri=model_uri,
                num_errors=len(errors),
                errors=error_details,
            )
            raise DependencyValidationError(
                f"Dependency validation failed for model '{model_uri}'. Mismatches found.",
                errors=errors,
            )

        logger.info("Dependency validation successful", model_uri=model_uri)

    def _get_dependencies_from_mlmodel(
        self, mlmodel: dict, model_uri: str
    ) -> dict[str, str]:
        """Extracts pip dependencies from conda.yaml or requirements.txt."""
        flavors = mlmodel.get("flavors", {})
        python_flavor = flavors.get("python_function", {})
        env_config = python_flavor.get("env", {})

        # Determine env file path (conda.yaml or requirements.txt)
        if "conda" in env_config:
            env_path = env_config["conda"]
        elif "virtualenv" in env_config:
            env_path = env_config["virtualenv"]
        else:
            logger.warning(
                "No conda or virtualenv env found in MLmodel", model_uri=model_uri
            )
            return {}

        try:
            deps_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"{model_uri}/{env_path}"
            )

            if env_path.endswith(".txt"):
                return self._parse_requirements_txt(deps_path)
            else:
                return self._parse_conda_yaml(deps_path)

        except Exception as e:
            logger.warning(
                "Could not read or parse environment file", path=env_path, exc_info=e
            )
            return {}

    def _parse_conda_yaml(self, file_path: str) -> dict[str, str]:
        """Parses pip dependencies from a conda.yaml file."""
        with open(file_path) as f:
            env_data = yaml.safe_load(f)

        pip_deps = {}
        for dep in env_data.get("dependencies", []):
            if isinstance(dep, dict) and "pip" in dep:
                for item in dep["pip"]:
                    lib, spec = self._parse_requirement_line(item)
                    if lib:
                        pip_deps[lib] = spec
        return pip_deps

    def _parse_requirements_txt(self, file_path: str) -> dict[str, str]:
        """Parses dependencies from a requirements.txt file."""
        pip_deps = {}
        with open(file_path) as f:
            for line in f:
                lib, spec = self._parse_requirement_line(line)
                if lib:
                    pip_deps[lib] = spec
        return pip_deps

    def _parse_requirement_line(self, line: str) -> tuple[str | None, str]:
        """Normalizes a requirement string into a library and a specifier."""
        line = line.strip()
        if not line or line.startswith("#"):
            return None, ""

        parts = (
            line.replace("==", "=")
            .replace("~=", "=")
            .replace(">=", "=")
            .replace("<=", "=")
            .split("=")
        )
        lib = parts[0].strip()
        spec = f"=={parts[1].strip()}" if len(parts) > 1 else ""
        return lib, spec

    def _find_required_spec(self, lib: str, dependencies: dict[str, str]) -> str | None:
        """Finds the version specifier for a library."""
        return dependencies.get(lib)

    def _is_version_valid(
        self,
        required_spec_str: str,
        installed_version: Version,
        rule: Literal["exact", "compatible"],
    ) -> bool:
        """Checks if the installed version satisfies the required specifier based on the rule."""
        logger.debug(
            "Comparing versions",
            library=rule,
            required_spec=required_spec_str,
            installed=str(installed_version),
            rule=rule,
        )
        if not required_spec_str:
            logger.warning("No required version specifier found, skipping validation.")
            return True

        spec = SpecifierSet(required_spec_str)

        if len(list(spec)) != 1 or next(iter(spec)).operator not in ["==", "==="]:
            logger.warning(
                "Cannot validate complex specifier, skipping", spec=str(spec)
            )
            return True

        required_version = Version(next(iter(spec)).version)

        is_valid = False
        if rule == "exact":
            is_valid = installed_version == required_version
        elif rule == "compatible":
            is_valid = (
                installed_version.major == required_version.major
                and installed_version.minor == required_version.minor
            )

        logger.debug("Validation result", is_valid=is_valid)
        return is_valid

    def _check_for_gpu_flavors(self, mlmodel: dict):
        """
        Checks for GPU-specific flavors in the MLmodel file.
        Raises DependencyValidationError if a GPU requirement is found in a CPU-only runtime.
        """
        flavors = mlmodel.get("flavors", {})
        has_gpu_flavor = (
            "gputest" in flavors
            or mlmodel.get("cuda_version") is not None
            or flavors.get("tensorflow", {}).get("gpu") is True
            or "onnx" in str(flavors)
            and "cuda" in str(flavors.get("onnx"))
        )

        if has_gpu_flavor:
            error = DependencyError(
                library="runtime",
                required="cpu",
                running="gpu",
                policy="exact",
            )
            logger.error(
                "GPU model flavor detected in a CPU-only runtime", mlmodel=mlmodel
            )
            raise DependencyValidationError(
                "Model requires GPU but service runs on CPU.", errors=[error]
            )
