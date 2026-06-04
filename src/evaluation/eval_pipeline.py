# src/evaluation/eval_pipeline.py
# CI/CD-la indha tests pass aana mattum dhaan deploy aagum

import json


def evaluate_rag(pipeline, test_questions: list) -> dict:
    """
    # Test questions kuduthu pipeline-ai evaluate pannrom
    #
    # Metrics:
    # - Answer Relevancy: Question-ku answer relevant-aa irukka?
    # - Citation Rate: evlo % answers-la citation irukku?
    # - Response Time: evlo neram aagudhu?
    """

    import time

    results = []
    citation_count = 0
    total_time = 0

    for qa_pair in test_questions:
        question = qa_pair["question"]
        expected_keywords = qa_pair.get("expected_keywords", [])

        start_time = time.time()
        result = pipeline.query(question)
        end_time = time.time()

        response_time = end_time - start_time
        total_time += response_time

        # Citation check
        has_citation = "[Source:" in result["answer"] or "Source:" in result["answer"]
        if has_citation:
            citation_count += 1

        # Keyword check - expected keywords answer-la irukka?
        keyword_hits = sum(
            1 for kw in expected_keywords
            if kw.lower() in result["answer"].lower()
        )
        keyword_score = keyword_hits / len(expected_keywords) if expected_keywords else 1.0

        results.append({
            "question": question,
            "answer": result["answer"],
            "has_citation": has_citation,
            "keyword_score": keyword_score,
            "response_time": response_time
        })

    # Summary metrics
    total = len(test_questions)
    metrics = {
        "total_questions": total,
        "citation_rate": citation_count / total,  # >= 0.9 venum
        "avg_keyword_score": sum(r["keyword_score"] for r in results) / total,
        "avg_response_time": total_time / total,  # < 10 seconds venum
        "results": results
    }

    return metrics


# Test questions - idha un domain-ku maathiko
TEST_QUESTIONS = [
    {
        "question": "What is the annual leave policy?",
        "expected_keywords": ["leave", "days", "annual"]
    },
    {
        "question": "How to apply for sick leave?",
        "expected_keywords": ["sick", "apply", "form"]
    }
]

if __name__ == "__main__":
    from src.pipeline import RAGPipeline

    print("🧪 Evaluation pannrom...")
    pipeline = RAGPipeline()
    pipeline.setup()

    metrics = evaluate_rag(pipeline, TEST_QUESTIONS)

    print("\n📊 Evaluation Results:")
    print(f"Citation Rate: {metrics['citation_rate']:.1%}")
    print(f"Avg Keyword Score: {metrics['avg_keyword_score']:.1%}")
    print(f"Avg Response Time: {metrics['avg_response_time']:.2f}s")

    # Save results
    with open("eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # CI gate - indha thresholds fail aana pipeline fail aagum
    assert metrics["citation_rate"] >= 0.8, "❌ Citation rate too low!"
    assert metrics["avg_response_time"] < 30, "❌ Too slow!"

    print("\n✅ All checks passed!")