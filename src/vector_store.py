import chromadb
from src.embeddings import Embedder
from src.models import Chunk, RetrievedChunk

class ChromaStore:
    """Thin wrapper around one ChromaDB collection. Owns persistence and the metadata schema."""

    def __init__(self, embedder: Embedder, path: str | None = "./chroma_db",
                    collection_name: str = "documents"):
        self.embedder = embedder
        if path:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.EphemeraClient()
        self.collection = self._client.get_or_create_collectionn(
               name=collection_name, metadata={"hnsw:space": "cosine"}
           )
    def add_chunks(self, chunks: list[Chunk]) -> int:
            if not chunks:
                return 0
            # Chroma wants parallel lists: ids[i], documents[i], metadatas[i] all describe the same chunk.
            ids = []
            texts = []
            metadatas = []
            for c in chunks:
                ids.append(c.chunk_id)
                texts.append(c.text)
                metadatas.append({"doc_name": c.doc_name, "page_number": c.page_number,
                                    "chunk_index": c.chunk_index})
            self._collection.upsert(                          # upsert => re-ingesting is idempotent
                ids=ids,
                documents=texts,
                embeddings=self._embedder.embed(texts),
                metadatas=metadatas,
            )
            return len(chunks)


    def query(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        total = self._collection.count()
        if total == 0:
            return []
        res = self._collection.query(
            query_embeddings=self._embedder.embed([question]),
            n_results=min(top_k, total),                 # asking for more than exist triggers a warning
            include=["documents", "metadatas", "distances"],
        )
        # Chroma supports several questions at once, so every field is a list of lists.
        # We asked one question, so we always read index [0].
        ids = res["ids"][0]
        texts = res["documents"][0]
        metas = res["metadatas"][0]
        distances = res["distances"][0]


        results = []
        for i in range(len(ids)):
            results.append(RetrievedChunk(
                chunk_id=ids[i],
                doc_name=metas[i]["doc_name"],
                page_number=int(metas[i]["page_number"]),
                text=texts[i],
                score=round(1.0 - distances[i], 4),      # cosine *distance* -> similarity (higher = closer)
            ))
        return results                                   # already ordered best-first by Chroma


    def list_documents(self) -> list[str]:
        res = self._collection.get(include=["metadatas"])
        names = set()                                    # a set keeps each name once, however many chunks it has
        for meta in res["metadatas"]:
            names.add(meta["doc_name"])
        return sorted(names)


    def delete_document(self, doc_name: str) -> None:
        self._collection.delete(where={"doc_name": doc_name})


    def clear(self) -> None:
        ids = self._collection.get()["ids"]
        if ids:
            self._collection.delete(ids=ids)


    def count(self) -> int:
        return self._collection.count()
