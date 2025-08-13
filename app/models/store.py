from threading import RLock
from typing import Mapping

from app.core.logging import get_logger
from app.models.loader import ModelBundle

logger = get_logger(__name__)


class ModelStore:
    """
    A thread-safe, in-memory store for loaded model bundles.
    Provides atomic operations for replacing the entire set of models.
    """

    def __init__(self):
        self._lock = RLock()
        self._store: Mapping[str, ModelBundle] = {}

    def get(self, key: str) -> ModelBundle | None:
        """
        Retrieves a model bundle by its key (e.g., "lightgbm").
        Returns None if the key is not found.
        """
        with self._lock:
            return self._store.get(key)

    def snapshot(self) -> Mapping[str, ModelBundle]:
        """
        Returns a shallow copy of the current store.
        This is safe to iterate over without a lock.
        """
        with self._lock:
            return self._store.copy()

    def replace_all(self, bundles: Mapping[str, ModelBundle]):
        """
        Atomically replaces the entire store with a new set of model bundles.
        This is the core of the hot-swap mechanism.
        """
        logger.info("Replacing all models in the store.")
        with self._lock:
            self._store = bundles
        logger.info("Model store replaced successfully.")


# A global instance of the model store that the application will use.
model_store = ModelStore()
