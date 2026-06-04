# src/retrieval/hybrid_search.py

# Hybrid Search ennanna:
# BM25 + Vector Search = irandum serthu use pannrom
#
# Vector Search: "vali" search pannaa "pain", "ache"-aiyum kaattum (semantic)
# BM25 Search: exact keywords match pannum
#
# Irandum serthaa - best of both worlds! 💪

from src.retrieval.vector_store import vector_search
from src.retrieval.bm25_search import BM25Searcher


def hybrid_search(vector_store, bm25_searcher: BM25Searcher,
                  query: str, top_k: int = 10,
                  vector_weight: float = 0.6,
                  bm25_weight: float = 0.4) -> list:
    """
    # Vector search + BM25 results-ai combine pannrom
    # vector_weight=0.6 → vector results-ku 60% importance
    # bm25_weight=0.4  → bm25 results-ku 40% importance
    """

    # 1. Vector search pannu
    vector_results = vector_search(vector_store, query, top_k=top_k)

    # 2. BM25 search pannu
    bm25_results = bm25_searcher.search(query, top_k=top_k)

    # 3. Results-ai combine pannuradhukku oru dictionary use pannrom
    combined_scores = {}  # chunk_id → score mapping
    chunk_map = {}  # chunk_id → actual chunk mapping

    # Vector results process pannu
    for rank, (doc, score) in enumerate(vector_results):
        # Chunk-oda unique id undaakku (content-ai key-aa use pannrom)
        chunk_id = doc.page_content[:100]  # First 100 chars as ID

        # Rank-based score: 1st result → highest score
        # RRF (Reciprocal Rank Fusion) formula use pannrom
        rrf_score = vector_weight * (1 / (rank + 60))

        combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + rrf_score
        chunk_map[chunk_id] = doc

    # BM25 results process pannu
    for rank, result in enumerate(bm25_results):
        chunk_id = result["chunk"].page_content[:100]

        rrf_score = bm25_weight * (1 / (rank + 60))

        combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + rrf_score
        chunk_map[chunk_id] = result["chunk"]

    # Score-oda order-la sort pannu (highest first)
    sorted_results = sorted(combined_scores.items(),
                            key=lambda x: x[1],
                            reverse=True)[:top_k]

    # Final results list return pannu
    final_results = []
    for chunk_id, score in sorted_results:
        final_results.append({
            "chunk": chunk_map[chunk_id],
            "hybrid_score": score
        })

    print(f"Hybrid search complete: {len(final_results)} results")
    return final_results