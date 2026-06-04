# src/reranking/reranker.py

# Reranking ennanna:
#
# Hybrid search-la 10 results varum
# Aana ellame equally good illa
#
# Cross-Encoder: Question + Answer pair-ai serthu read panni
#                "indha answer indha question-ku evlo relevant?"
#                nu score pannum
#
# Idhu slow but very accurate! ⚡

from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        # Cross-encoder model load pannrom
        # First time-la download aagum (~80MB)
        """
        print(f"Reranker model load pannrom: {model_name}")
        self.model = CrossEncoder(model_name)
        print("Reranker ready!")

    def rerank(self, query: str, results: list, top_k: int = 5) -> list:
        """
        # Query + each chunk pair-ai score panni best-ai select pannrom
        """

        if not results:
            return []

        # Query + chunk pairs undaakku
        # [(query, chunk1), (query, chunk2), ...]
        pairs = [
            (query, result["chunk"].page_content)
            for result in results
        ]

        # Cross-encoder scores calculate pannu
        # Idhu ovvoru pair-aiyum deeply analyze pannum
        scores = self.model.predict(pairs)

        # Results-ku scores add pannu
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        # Rerank score-oda order-la sort pannu
        reranked = sorted(results,
                          key=lambda x: x["rerank_score"],
                          reverse=True)[:top_k]

        print(f"Reranking done: Top {len(reranked)} results selected")
        return reranked