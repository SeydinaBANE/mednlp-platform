"""Abstract base class for all embedding models."""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Common interface for all embedding models in the platform."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier (used as Qdrant collection prefix)."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Version string (e.g. 'v1', '2024-03')."""

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Output dimensionality of the embedding vectors."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text."""

    async def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for a single text."""
        vectors = await self.embed([text])
        return vectors[0]
