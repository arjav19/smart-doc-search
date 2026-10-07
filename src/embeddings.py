from typing import Protocol

class Embedder(Protocol):
    """A Protocol is just a promise about method names — NOT a parent class.

    Any object with an `embed(texts)` method counts as an Embedder; no inheritance needed.
    That is exactly what lets the tests swap in `FakeEmbedder` (tests/fakes.py) without
    the real model, and what would let us add a GeminiEmbedder later without touching
    ChromaStore. The `...` body means "no implementation here, only the signature".
    """
    
    def embed(self, text: list[str]) -> list[list[float]]: ...

class LocalEmbedder:
    """Free, offline embeddings via sentence-transformers (all-MiniLM-L6-v2, 384 dims)."""


    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer   # lazy: heavy import, keeps tests fast
        self._model = SentenceTransformer(model_name)            # first call downloads ~90 MB, then cached


    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
           return []
        # normalize_embeddings=True -> every vector has length 1, so dot product == cosine similarity.
        # .tolist() converts the NumPy array into plain Python lists (what Chroma expects).
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()
