# src/retrieval/vector_store.py

# Vector Store ennanna:
# "Hello" → [0.2, 0.8, 0.1, 0.9, ...] (numbers-aa maathurom)
# Similar meaning-aa irundha similar numbers irukkum
# Idha vachu semantic search pannalaam

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

load_dotenv()  # .env file-la irukkura API key load pannu


def create_vector_store(chunks: list, persist_directory: str = "./chroma_db"):
    """
    # Chunks-ai vectors-aa maathi ChromaDB-la save pannrom
    # Idhu oru time mattum panna vendiyadhu
    """

    print("Embeddings undaakkrom... (konjam neram aagum)")

    # HuggingFace local embedding model use pannrom
    # Idhu text-ai numbers-aa (vectors) maathum - complete-aa local and free!
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ChromaDB-la ella chunks-aiyum store pannu
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory  # Disk-la save aagum
    )

    print(f"Vector store ready! {persist_directory}-la save aachu")
    return vector_store


def load_vector_store(persist_directory: str = "./chroma_db"):
    """
    # Already save aana vector store-ai load pannrom
    # Ovvoru time-um recreate panna vendaam
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    return vector_store


def vector_search(vector_store, query: str, top_k: int = 5) -> list:
    """
    # Query-kku similar chunks-ai vector store-la thedurom
    # Semantic search - meaning-based search
    """

    # top_k = 5 → 5 most similar chunks return pannum
    results = vector_store.similarity_search_with_score(query, k=top_k)

    return results