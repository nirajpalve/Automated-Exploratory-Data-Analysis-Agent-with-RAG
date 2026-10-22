import re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

KNOWLEDGE_FILE = Path("knowledge/eda_best_practices.md")
DB_PATH = "chroma_db"
COLLECTION = "eda_practices"

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path=DB_PATH)


def build_index():
    """Split the best-practices doc into sections and store them in ChromaDB."""
    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")

    # Split before every line that starts with '## ' and keep only real sections
    sections = re.split(r"\n(?=## )", text)
    chunks = [s.strip() for s in sections if s.strip().startswith("## ")]
    titles = [c.splitlines()[0].replace("## ", "").strip() for c in chunks]

    # Start fresh each time so edited or removed sections don't linger
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, embedding_function=embedder)

    col.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"title": t} for t in titles],
    )
    print(f"Indexed {len(chunks)} sections:")
    for t in titles:
        print("  -", t)


def retrieve(query: str, k: int = 3) -> list[str]:
    """Return the k sections most relevant to the query."""
    col = client.get_collection(COLLECTION, embedding_function=embedder)
    result = col.query(query_texts=[query], n_results=k)
    return result["documents"][0]


if __name__ == "__main__":
    build_index()

    print("\nTesting retrieval:")
    tests = [
        "a column has 60% missing values",
        "income is heavily right-skewed",
        "two features are almost perfectly correlated",
        "a column has 5000 unique text values",
    ]
    for q in tests:
        top = retrieve(q, k=1)[0]
        print(f"  '{q}'  ->  {top.splitlines()[0]}")