from abc import ABC, abstractmethod
from typing import Any

class BaseModelClient(ABC):
    @abstractmethod
    def generate_structured_json(self, prompt: str, json_schema: Any) -> Any:
        raise NotImplementedError