from typing import Callable, Dict


class ModelRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str, factory: Callable) -> None:
        if name in self._registry:
            raise ValueError(f"Model '{name}' is already registered")
        self._registry[name] = factory

    def get(self, name: str) -> Callable:
        if name not in self._registry:
            raise KeyError(f"Unknown model '{name}'. Registered: {list(self._registry)}")
        return self._registry[name]


MODEL_REGISTRY = ModelRegistry()
