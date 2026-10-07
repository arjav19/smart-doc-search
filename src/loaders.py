from pathlib import Path
from typing import BinaryIO
from pypdf import PdfReader
from src.exceptions import NoTextLayerError,UnsupportedFileError
from src.models import Page

class PDFLoder:
    """Extracts page-aware text from a file path or an uploaded file object."""

    @staticmethod
    def load(source: str| Path|BinaryIO , doc_name:str| None= None)->list[Page]:
        # Work out the display name. Three kinds of `source` can arrive here:
        #   - a path string  "data/sample_pdfs/handbook.pdf"   (CLI)
        #   - a Path object  Path("data/.../handbook.pdf")      (tests)
        #   - an uploaded file object with a .name attribute    (Streamlit)
        if doc_name is None:
            raw_name = getattr(source, "name", None)
            if raw_name is None:
                raw_name = str(source)
            doc_name = Path(raw_name).name    
        name =  doc_name
        if not name.lower().endswith(".pdf"):
            raise UnsupportedFileError(f"Only .pdf files are supported, got: {name}")
        reader =  PdfReader(source)
        pages = []
        for number,page in enumerate(reader.pages,start=1):
            text = (pages.extrac_text() or "").strip()
            if text:
                pages.append(Page(doc_name=name,page_number=number,text=text))
        if not pages:
            raise NoTextLayerError(
                   f"'{name}' has no extractable text. Is it a scanned PDF? OCR is not supported."
               )
        return pages
        