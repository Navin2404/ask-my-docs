# src/generation/rag_chain.py

# Idhu dhaan main part!
# Retrieved chunks-ai LLM-ku kuduthu answer generate pannrom
# IMPORTANT: Citation enforce pannrom - "Source: X" illaama answer varakoodaadhu

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Citation enforce panna System Prompt
# LLM-ku strict instructions kudukrom
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context documents.

STRICT RULES:
1. Answer ONLY from the given context - never use outside knowledge
2. Every claim must have a citation in format: [Source: filename, Page: X]  
3. If answer is not in context, say: "I cannot find this information in the provided documents"
4. Be concise and accurate

Context Documents:
{context}
"""


def format_context_with_citations(reranked_results: list) -> tuple:
    """
    # Reranked results-ai LLM-ku kudukka format pannrom
    # Ovvoru chunk-kum source information add pannrom
    """

    formatted_chunks = []
    citation_map = {}  # Citation tracking

    for i, result in enumerate(reranked_results):
        chunk = result["chunk"]

        # Source information edu
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", "N/A")

        # File name mattum edu (full path venam)
        filename = os.path.basename(source)

        # Formatted chunk create pannu
        chunk_text = f"""[Document {i + 1}]
Source: {filename}, Page: {page}
Content: {chunk.page_content}
"""
        formatted_chunks.append(chunk_text)
        citation_map[i + 1] = {"filename": filename, "page": page}

    context = "\n---\n".join(formatted_chunks)
    return context, citation_map


def generate_answer(query: str, reranked_results: list) -> dict:
    """
    # Main function: Query + Context → Answer with Citations
    """

    # Context format pannu
    context, citation_map = format_context_with_citations(reranked_results)

    # Groq-oda LLM initialize pannu
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # Fast and highly capable model on Groq
        temperature=0,  # 0 = deterministic, facts-based answer
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Prompt template create pannu
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    # Chain: Prompt → LLM
    chain = prompt | llm

    # Answer generate pannu
    print(f"Answer generate pannrom...")
    response = chain.invoke({
        "context": context,
        "question": query
    })

    return {
        "answer": response.content,
        "citations": citation_map,
        "num_sources": len(reranked_results)
    }