from dataclasses import dataclass,field
# @dataclass writes __init__, __repr__ and __eq__ for you from the field list below.
   # frozen=True makes instances read-only (page.text = "x" raises an error) — safe to pass around.

@dataclass(frozen=True)
class Page:
    doc_name : str
    page_number : int
    text : str

@dataclass(frozen=True)
class Chunk:
    chunk_id:str
    doc_name : str
    doc_number : int
    chunk_index:int
    text : str

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id:str
    doc_name : str
    page_number : int
    text : str
    score : float         

@dataclass(frozen=True)
class Citetion:
    marker:int
    doc_name : str
    page_number : int
    snippet : str
    score : float         

@dataclass
class RAGResponse:
    answer : str
    citetion : list[Citetion]
    grounded: bool              # True if claims are backed by verified citations, or the model honestly abstained
    # Python forbids `= []` as a default (one shared list for every instance!); default_factory=list
    # means "call list() to make a fresh empty list for each new RAGResponse".
    retrieved: list[RetrievedChunk] = field(default_factory=list)
