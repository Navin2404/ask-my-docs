User Question
      ↓
┌─────────────────────────────────────────┐
│           HYBRID RETRIEVAL              │
│                                         │
│   BM25 Search     +    Vector Search    │
│  (Keyword-based)     (Semantic/Meaning) │
│         └──────────┬────────────────┘  │
│                    ↓                    │
│         Cross-Encoder Reranking         │
│      (Picks the most relevant chunks)   │
└─────────────────────────────────────────┘
      ↓
  Top 5 Relevant Chunks
      ↓
  GPT-3.5 Turbo → Answer + Citations
      ↓
  "The leave policy states 25 days per year. [Source: leave_policy.txt, Page: 1]"


Project Structure
ask-my-docs/
│
├── data/                          # Your documents go here
│   ├── leave_policy.txt
│   ├── employee_handbook.txt
│   ├── it_support_policy.txt
│   └── performance_review.txt
│
├── src/
│   ├── ingestion/
│   │   └── document_loader.py     # PDF & TXT loading + chunking
│   │
│   ├── retrieval/
│   │   ├── vector_store.py        # ChromaDB vector search
│   │   ├── bm25_search.py         # BM25 keyword search
│   │   └── hybrid_search.py       # Combines both searches (RRF)
│   │
│   ├── reranking/
│   │   └── reranker.py            # Cross-encoder reranking
│   │
│   ├── generation/
│   │   └── rag_chain.py           # LLM answer generation with citations
│   │
│   ├── evaluation/
│   │   └── eval_pipeline.py       # Automated quality evaluation
│   │
│   └── pipeline.py                # Master pipeline connecting all modules
│
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── .gitignore
└── .github/
    └── workflows/
        └── eval_ci.yml            # CI/CD pipeline