"""Day 1 experiment: do embeddings capture meaning,not just shared words?"""
from sentence_transformers import SentenceTransformer

SENTENCES = [
    "The cat sat on the mat.",
    "A kitten is resting on a rug.",
    "Python is a popular programming language.",
    "Django is a web framework written in Python.",
    "The stock market fell sharply today.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")        # ~90 MB, downloaded once to ~/.cache
emb = model.encode(SENTENCES, normalize_embeddings=True)  # NumPy array, shape (5, 384): one row per sentence
# `@` is NumPy's matrix-multiply operator and `.T` is the transpose, so emb @ emb.T is a 5x5 table
# where cell [i][j] = dot product of sentence i and sentence j. Rows have length 1 (normalized),
# so each dot product IS the cosine similarity: 1.0 = identical meaning, ~0 = unrelated.
sim = emb @ emb.T


print(f"Embedding shape: {emb.shape}\n")
for i, s in enumerate(SENTENCES):
   print(f"[{i}] {s}")


print("\nCosine similarity matrix:")
header = "      "
for j in range(len(SENTENCES)):
   header += f"  [{j}]"
print(header)
for i, row in enumerate(sim):
    line = f"[{i}]   "
    for value in row:
       line += f"{value:5.2f} "                           # 5 characters wide, 2 decimals
    print(line)
