from typing import Protocol


class SQLModel(Protocol):
    name: str

    def generate(self, prompt: str) -> str:
        ...
