Ask My Docs — Production RAG Application
Upload your documents and ask questions in natural language. Get accurate answers with source citations.
What it does

Upload PDF or TXT files
Ask questions about your documents
Get answers with exact source citations (filename + page number)
AI answers only from your documents — no random hallucination

Tech Stack

LLM — OpenAI GPT-3.5 Turbo
Vector Database — ChromaDB
Search — Hybrid Search (BM25 + Vector Search)
Reranking — Cross-Encoder (sentence-transformers)
Framework — LangChain
UI — Streamlit
CI/CD — GitHub Actions
Language — Python 3.11

Key Concepts

BM25 Search — Keyword-based search (finds exact word matches)
Vector Search — Semantic search (finds meaning-based matches)
Hybrid Search — Combines both for better accuracy
Reranking — Picks the most relevant results from the combined search
Citation Enforcement — Every answer includes the source document and page number