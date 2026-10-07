"""Usage: python -m scripts.ingest data/sample_pdfs/*.pdf"""
import sys
from src.chunker import TextChunker
from src.config import Settings
from src.embeddings import LocalEmbedder
from src.exceptions import SmartDocSearchError
from src.loaders import PDFLoder
from src.vector_store import ChromaStore

def main(paths : list[str]) -> None:
    settings =  Settings.from_env()
    store = ChromaStore(LocalEmbedder(settings.embedding_model), path=settings.chroma_path)
    chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
    for path in paths:
        try:
            pages = PDFLoder.load(path)
            added = store.add_chunks(chunker.chunk_pages(pages))
            print(f"{path}:{len(pages)} pages -> {added} chunks")
        except SmartDocSearchError as e:
            print(f"{path}: SKIPPED ({e})", file=sys.stderr)
    print(f"Collection now holds {store.count()} chunks")


if __name__ == "__main__":
    if len(sys.argv) < 2:
       sys.exit("Usage: python -m scripts.ingest <file.pdf> [more.pdf ...]")
    main(sys.argv[1:])
