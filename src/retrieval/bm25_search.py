# src/retrieval/bm25_search.py

# BM25 ennanna:
# Traditional keyword-based search (Google maadhiri)
# "leave policy" search pannaa - andha exact words irukkura documents thedum
# Vector search miss pannura specific terms-ai idhu catch pannum

from rank_bm25 import BM25Okapi
import numpy as np


class BM25Searcher:

    def __init__(self, chunks: list):
        """
        # Chunks-ai vachu BM25 index undaakkrom
        """

        self.chunks = chunks  # Original chunks save pannu

        # Ovvoru chunk-aiyum words-aa split pannu
        # "hello world" → ["hello", "world"]
        tokenized_corpus = [
            chunk.page_content.lower().split()
            for chunk in chunks
        ]

        # BM25 index undaakku
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index ready! {len(chunks)} chunks indexed")

    def search(self, query: str, top_k: int = 5) -> list:
        """
        # Query-kku matching chunks-ai keyword-based-aa thedu
        """

        # Query-aiyum words-aa split pannu
        tokenized_query = query.lower().split()

        # BM25 scores calculate pannu
        scores = self.bm25.get_scores(tokenized_query)

        # Top K highest scores-oda indices edu
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Results list undaakku
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Score 0-ai vida adhigamaa irundhaa mattum
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(scores[idx]),
                    "index": int(idx)
                })

        return results