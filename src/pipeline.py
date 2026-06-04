# src/pipeline.py
# Ellaathayum oru place-la connect pannrom

from src.ingestion.document_loader import load_documents, split_documents
from src.retrieval.vector_store import create_vector_store, load_vector_store
from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.hybrid_search import hybrid_search
from src.reranking.reranker import Reranker
from src.generation.rag_chain import generate_answer
import os


class RAGPipeline:

    def __init__(self, data_folder: str = "./data",
                 vector_db_path: str = "./chroma_db"):

        self.data_folder = data_folder
        self.vector_db_path = vector_db_path
        self.chunks = None
        self.vector_store = None
        self.bm25_searcher = None
        self.reranker = None

    def setup(self, force_reload: bool = False):
        """
        # Pipeline-ai oru time setup pannu
        # force_reload=True → documents change aana re-index pannu
        """

        # Documents load & split
        print("📄 Documents load pannrom...")
        documents = load_documents(self.data_folder)
        self.chunks = split_documents(documents)

        # Vector store - already irundha load, illena create
        if os.path.exists(self.vector_db_path) and not force_reload:
            print("💾 Existing vector store load pannrom...")
            self.vector_store = load_vector_store(self.vector_db_path)
        else:
            print("🔢 Pudhusa vector store create pannrom...")
            self.vector_store = create_vector_store(
                self.chunks, self.vector_db_path
            )

        # BM25 searcher
        print("🔍 BM25 index create pannrom...")
        self.bm25_searcher = BM25Searcher(self.chunks)

        # Reranker
        print("🏆 Reranker load pannrom...")
        self.reranker = Reranker()

        print("\n✅ RAG Pipeline ready!")

    def query(self, question: str) -> dict:
        """
        # User question process panni answer return pannrom
        # Main flow:
        # Question → Hybrid Search → Rerank → Generate Answer
        """

        print(f"\n❓ Question: {question}")

        # Step 1: Hybrid Search (BM25 + Vector)
        print("🔎 Hybrid search pannrom...")
        hybrid_results = hybrid_search(
            self.vector_store,
            self.bm25_searcher,
            question,
            top_k=10  # Top 10 results edu
        )

        # Step 2: Rerank - Best 5 select pannu
        print("🏆 Reranking pannrom...")
        reranked = self.reranker.rerank(question, hybrid_results, top_k=5)

        # Step 3: Answer Generate with Citations
        print("💬 Answer generate pannrom...")
        result = generate_answer(question, reranked)

        return result