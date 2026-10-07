import hashlib
import re 
from src.models import Chunk,Page

# A paragraph boundary is a blank line:  newline, then optional spaces/tabs, then another newline.
#   \n   newline        \s*   zero or more whitespace characters        \n   newline
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


class TextChunker:
    """Packs paragraphs into chunks of at most `chunk_size` characters. Never crosses a page."""

    def __init__(self, chunk_size: int = 1200, overlap: int = 150):
        if overlap >= chunk_size:
           raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlape = overlap

    def chunk_pages(self, pages: list[Page]) -> list[Chunk]:
        chunks : list[Chunk] = []
        for page in pages:
           for piece in self._split_text(page.text):
              index = len(chunks)
              chunks.append(Chunk(
                 chunk_id=self._make_id(page.doc_name, page.page_number, index),
                       doc_name=page.doc_name,
                       page_number=page.page_number,
                       chunk_index=index,
                       text=piece,
              ))
        return chunks



    def _split_text(self, text: str) -> list[str]:
        """Split one page into pieces of at most chunk_size characters, keeping paragraphs whole.


        Worked example with chunk_size=1200 and three paragraphs of 500 chars each:


            - p1 (500) fits in the empty buffer                     -> buffer = p1
            - p2 (500): 500 + 500 + 1 = 1001 <= 1200, still fits    -> buffer = p1 + "\\n" + p2
            - p3 (500): 1001 + 500 + 1 = 1502 > 1200, does NOT fit  -> flush buffer, buffer = p3
            - end of page                                            -> flush p3
            Result: ["p1\\np2", "p3"]
        A single paragraph bigger than chunk_size is cut with a sliding window instead.
        """
        # 1. Split on blank lines and drop empty leftovers.
        paragraphs = []
        for raw in _PARAGRAPH_SPLIT.split(text):
            para = raw.strip()
            if para:
              paragraphs.append(para)

        # 2. Pack paragraphs into a buffer until the next one would overflow chunk_size.
            pieces: list[str] = []
            buffer = ""
            for para in paragraphs:
                if len(para) > self.chunk_size:                  # giant paragraph -> hard split
                    if buffer:
                        pieces.append(buffer)
                        buffer = ""
                    pieces.extend(self._sliding_window(para))
                elif len(buffer) + len(para) + 1 <= self.chunk_size:   # +1 for the "\n" joiner; fits -> keep packing
                    if buffer:
                        buffer = buffer + "\n" + para
                    else:
                        buffer = para
                else:                                             # doesn't fit -> flush, start new
                    if buffer:
                        pieces.append(buffer)
                    buffer = para
            if buffer:                                            # whatever is left at the end of the page
                pieces.append(buffer)
            return pieces


    def _sliding_window(self, text: str) -> list[str]:
        """Cut a huge paragraph into windows that overlap by `overlap` chars so no sentence is lost at a cut."""
        step = self.chunk_size - self.overlap
        pieces = []
        for start in range(0, len(text), step):
            pieces.append(text[start:start + self.chunk_size])
            if start + self.chunk_size >= len(text):      # stop once a window reaches the end
                break
        return pieces


    @staticmethod
    def _make_id(doc_name: str, page: int, index: int) -> str:
            # Same (doc, page, index) always gives the same id -> re-ingesting a file overwrites instead of duplicating.
            # sha1(...) needs bytes, hence .encode(); hexdigest() is 40 hex chars, we keep the first 16.
            return hashlib.sha1(f"{doc_name}:{page}:{index}".encode()).hexdigest()[:16]
        