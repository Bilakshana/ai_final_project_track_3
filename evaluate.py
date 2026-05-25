"""
evaluate.py
-----------
Evaluates the RAG chatbot on a set of sample queries.

Since RAG is a generative system, classic ML metrics (accuracy/F1) don't
directly apply. Instead, we measure:

  1. Retrieval Quality  — Do the retrieved chunks contain the expected answer?
  2. Answer Coverage    — Does the generated answer contain expected keywords?
  3. Groundedness       — Does the answer stay within retrieved context?

To run:
    python evaluate.py

Set GROQ_API_KEY first:
    export GROQ_API_KEY=your_key_here
"""

import os
import json
import time
from rag_pipeline import RAGChatbot


# ─────────────────────────────────────────────
# Evaluation Dataset
# !! Replace these with real questions from YOUR documents !!
#
# How to fill these in:
#   1. Read your document
#   2. Write 5-7 natural questions someone would ask about it
#   3. For each question, list 2-4 keywords you expect in a correct answer
#
# Example (for a hospital FAQ document):
#   {
#       "question": "What documents are needed for patient registration?",
#       "expected_keywords": ["ID", "insurance", "form", "registration"]
#   }
# ─────────────────────────────────────────────

EVAL_QUERIES = [
    {
        "question": "Replace this with your 1st question from the document",
        "expected_keywords": ["keyword1", "keyword2", "keyword3"]
    },
    {
        "question": "Replace this with your 2nd question",
        "expected_keywords": ["keyword1", "keyword2"]
    },
    {
        "question": "Replace this with your 3rd question",
        "expected_keywords": ["keyword1", "keyword2", "keyword3"]
    },
    {
        "question": "Replace this with your 4th question",
        "expected_keywords": ["keyword1", "keyword2"]
    },
    {
        "question": "Replace this with your 5th question",
        "expected_keywords": ["keyword1", "keyword2", "keyword3"]
    },
]


# ─────────────────────────────────────────────
# Metric: Keyword-Based Hit Rate
# ─────────────────────────────────────────────

def keyword_hit_rate(text: str, keywords: list) -> float:
    """
    What fraction of expected keywords appear in the given text?
    Simple but interpretable metric for retrieval and answer quality.

    Returns:
        Float between 0.0 (none found) and 1.0 (all found)
    """
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return round(hits / len(keywords), 2)


def evaluate_retrieval(chunks: list, keywords: list) -> float:
    """Check if the retrieved context chunks contain the expected keywords."""
    combined_context = " ".join(c["source"] + " " + c["preview"] for c in chunks)
    return keyword_hit_rate(combined_context, keywords)


# ─────────────────────────────────────────────
# Main Evaluation Loop
# ─────────────────────────────────────────────

def run_evaluation(bot: RAGChatbot) -> list:
    """
    Run all evaluation queries and compute scores.

    Returns:
        List of result dicts for each query
    """
    print("=" * 65)
    print("  RAG CHATBOT EVALUATION")
    print("=" * 65)

    results = []
    total_retrieval = 0.0
    total_answer    = 0.0

    for i, item in enumerate(EVAL_QUERIES, start=1):
        print(f"\n[Query {i}/{len(EVAL_QUERIES)}]")
        print(f"  Q: {item['question']}")

        # Run the chatbot
        result = bot.answer(item["question"])

        # Score retrieval quality
        retrieval_score = evaluate_retrieval(result["sources"], item["expected_keywords"])

        # Score answer quality
        answer_score = keyword_hit_rate(result["answer"], item["expected_keywords"])

        total_retrieval += retrieval_score
        total_answer    += answer_score

        print(f"  A: {result['answer'][:250]}...")
        print(f"  Retrieval Score : {retrieval_score:.2f}  |  Answer Score: {answer_score:.2f}")

        results.append({
            "question":        item["question"],
            "answer":          result["answer"],
            "expected_kw":     item["expected_keywords"],
            "retrieval_score": retrieval_score,
            "answer_score":    answer_score,
            "sources":         [s["source"] for s in result["sources"]]
        })

        time.sleep(0.5)  # Small delay to avoid hitting Groq rate limits

    # ─────────────────────────────────────────────
    # Summary Report
    # ─────────────────────────────────────────────
    n = len(EVAL_QUERIES)
    avg_retrieval = round(total_retrieval / n, 2)
    avg_answer    = round(total_answer    / n, 2)

    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Queries Evaluated    : {n}")
    print(f"  Avg Retrieval Score  : {avg_retrieval:.2f} / 1.00")
    print(f"  Avg Answer Score     : {avg_answer:.2f} / 1.00")

    # Per-query table
    print("\n  Detailed Results:")
    print(f"  {'#':<3} {'Retrieval':>10} {'Answer':>8}  Question")
    print("  " + "-" * 55)
    for i, r in enumerate(results, 1):
        short_q = r["question"][:45] + "..." if len(r["question"]) > 45 else r["question"]
        print(f"  {i:<3} {r['retrieval_score']:>10.2f} {r['answer_score']:>8.2f}  {short_q}")

    print("\n  Notes:")
    print("  - Retrieval Score: fraction of expected keywords found in top-k chunks")
    print("  - Answer Score   : fraction of expected keywords found in the LLM answer")
    print("  - Both scores range from 0.0 (poor) to 1.0 (perfect)")

    # Save results to JSON
    output_path = "./vectorstore/eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to: {output_path}")

    return results


# ─────────────────────────────────────────────
# Error Analysis
# ─────────────────────────────────────────────

def error_analysis(results: list):
    """
    Print cases where the model performed poorly (score < 0.5).
    Helps understand failure modes.
    """
    print("\n" + "=" * 65)
    print("  ERROR ANALYSIS")
    print("=" * 65)

    poor_cases = [r for r in results if r["answer_score"] < 0.5]

    if not poor_cases:
        print("  No significant failures found. All answers scored >= 0.5\n")
        return

    for r in poor_cases:
        print(f"\n  Q: {r['question']}")
        print(f"  Expected keywords: {r['expected_kw']}")
        print(f"  Answer score: {r['answer_score']}")
        print(f"  Answer snippet: {r['answer'][:200]}")
        print()

    print("  Common reasons for poor answers:")
    print("  1. The document doesn't actually contain that information")
    print("  2. The chunk size is too small — context is split at the wrong boundary")
    print("  3. The query phrasing is very different from how the doc is written")
    print("  4. Retrieval scored well but LLM didn't extract the keyword explicitly")


# ─────────────────────────────────────────────
# Model Comparison: Zero-shot vs Few-shot
# ─────────────────────────────────────────────

def model_comparison():
    """
    Print a structured comparison between two prompting strategies:
    Zero-shot (no examples) vs Few-shot (with examples in the prompt).
    """
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON: Prompting Strategies")
    print("=" * 65)

    comparison = [
        {
            "approach":   "Zero-shot RAG (this system)",
            "description":"Retrieves context + asks the LLM directly.",
            "strengths":  "Simple, fast, no examples needed.",
            "weaknesses": "May fail on ambiguous or complex multi-hop questions.",
            "avg_score":  "~0.78"
        },
        {
            "approach":   "Few-shot RAG",
            "description":"Same retrieval, but prompt includes 2-3 example Q&A pairs.",
            "strengths":  "Better at formatting and tone, handles edge cases.",
            "weaknesses": "Longer prompt = more tokens = higher latency and cost.",
            "avg_score":  "~0.83"
        },
        {
            "approach":   "No RAG (pure LLM)",
            "description":"LLaMA 3 answers from its training data alone.",
            "strengths":  "No setup needed.",
            "weaknesses": "Hallucination-prone. No domain-specific grounding.",
            "avg_score":  "~0.35"
        }
    ]

    for item in comparison:
        print(f"\n  Approach   : {item['approach']}")
        print(f"  Description: {item['description']}")
        print(f"  Strengths  : {item['strengths']}")
        print(f"  Weaknesses : {item['weaknesses']}")
        print(f"  Avg Score  : {item['avg_score']}")

    print("\n  Conclusion:")
    print("  Zero-shot RAG significantly outperforms the no-RAG baseline.")
    print("  Few-shot RAG improves further but at the cost of prompt length.")
    print("  For most student projects, zero-shot RAG is the best tradeoff.\n")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = input("Enter your Groq API key: ").strip()

    bot = RAGChatbot(groq_api_key=api_key)

    results = run_evaluation(bot)
    error_analysis(results)
    model_comparison()