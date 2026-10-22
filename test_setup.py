import pandas, seaborn, chromadb, streamlit
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding size:", len(model.encode("hello")))
print("Setup OK")
