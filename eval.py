#!/usr/bin/env python3
"""Evaluate RAG pipeline quality via LLM-as-judge.

Usage:
    python eval.py
    python eval.py --model "model-name"
    python eval.py --queries my_queries.json
"""
import argparse
import json
import re
import sys

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

import config
from rag import build_retriever


def _judge_llm():
    return ChatOpenAI(
        model=config.DEFAULT_CHAT_MODEL,
        base_url=config.LM_STUDIO_URL,
        api_key="not-needed",
        temperature=0,
    )


def _ask_judge(system: str, user: str, llm) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", user),
    ])
    result = (prompt | llm).invoke({})
    return result.content if hasattr(result, "content") else str(result)


def score_relevance(query: str, doc_text: str, llm) -> bool:
    system = (
        "You are evaluating document retrieval quality. "
        "Determine if the document is RELEVANT to answering the question. "
        "Answer only YES or NO."
    )
    user = f"Question: {query}\n\nDocument:\n{doc_text[:2000]}\n\nIs this document relevant? YES or NO:"
    ans = _ask_judge(system, user, llm).strip().upper()
    return ans.startswith("YES")


def score_faithfulness(query: str, answer: str, context: str, llm) -> float:
    system = (
        "You are evaluating whether an answer is faithful to the provided context. "
        "Rate from 0 to 10 where:\n"
        "0 = The answer contains information not supported by or contradicting the context\n"
        "10 = The answer is completely supported by the context with no unsupported claims\n\n"
        "Return only the number."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer: {answer}\n\nFaithfulness score (0-10):"
    ans = _ask_judge(system, user, llm).strip()
    m = re.search(r"(\d+(?:\.\d+)?)", ans)
    return float(m.group(1)) if m else 0.0


def score_completeness(query: str, answer: str, llm) -> float:
    system = (
        "You are evaluating whether an answer completely addresses the question. "
        "Rate from 0 to 10 where:\n"
        "0 = The answer does not address the question at all\n"
        "10 = The answer fully and thoroughly addresses all aspects of the question\n\n"
        "Return only the number."
    )
    user = f"Question: {query}\n\nAnswer: {answer}\n\nCompleteness score (0-10):"
    ans = _ask_judge(system, user, llm).strip()
    m = re.search(r"(\d+(?:\.\d+)?)", ans)
    return float(m.group(1)) if m else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline quality")
    parser.add_argument("--queries", default="test_queries.json",
                        help="Path to test queries JSON")
    parser.add_argument("--model", default=config.DEFAULT_CHAT_MODEL,
                        help="Chat model for generation (not judge)")
    args = parser.parse_args()

    with open(args.queries) as f:
        test_queries = json.load(f)
    print(f"Loaded {len(test_queries)} test queries from {args.queries}")

    print("Building retriever...")
    retriever = build_retriever()

    gen_llm = ChatOpenAI(
        model=args.model,
        base_url=config.LM_STUDIO_URL,
        api_key="not-needed",
        temperature=0.3,
    )
    judge = _judge_llm()

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant that answers questions based on "
        "the provided Wikipedia articles.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n\n"
        "Answer concisely and cite the article titles you used as sources."
    )
    gen_chain = prompt | gen_llm

    results = []
    for item in tqdm(test_queries, desc="Evaluating", unit="q"):
        q = item["query"]

        docs = retriever.invoke(q)
        context_text = "\n\n---\n\n".join(d.page_content for d in docs)

        answer = gen_chain.invoke({"context": docs, "input": q})
        answer_text = answer.content if hasattr(answer, "content") else str(answer)

        # Precision@K
        relevant_count = 0
        for d in docs:
            if score_relevance(q, d.page_content, judge):
                relevant_count += 1
        precision = relevant_count / len(docs) if docs else 0.0

        # Faithfulness
        faithfulness = score_faithfulness(q, answer_text, context_text, judge)

        # Completeness
        completeness = score_completeness(q, answer_text, judge)

        results.append({
            "query": q,
            "difficulty": item.get("difficulty", "?"),
            "precision": precision,
            "faithfulness": faithfulness,
            "completeness": completeness,
            "n_docs": len(docs),
        })

    # --- Report ---
    print("\n" + "=" * 60)
    print("RAG EVALUATION REPORT")
    print("=" * 60)

    precisions = np.array([r["precision"] for r in results])
    faithfulnesses = np.array([r["faithfulness"] for r in results])
    completenesses = np.array([r["completeness"] for r in results])

    print(f"\nOverall ({len(results)} queries):")
    print(f"  Precision@K:  {precisions.mean():.2f}  (per-query avg)")
    print(f"  Faithfulness: {faithfulnesses.mean():.2f} / 10")
    print(f"  Completeness: {completenesses.mean():.2f} / 10")

    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if not subset:
            continue
        p = np.mean([r["precision"] for r in subset])
        f = np.mean([r["faithfulness"] for r in subset])
        c = np.mean([r["completeness"] for r in subset])
        print(f"\n  {diff.capitalize():>8} ({len(subset)} q):  P={p:.2f}  F={f:.2f}  C={c:.2f}")

    print("\n" + "-" * 60)
    print(f"{'Query':<50} {'P':>5} {'F':>5} {'C':>5}")
    print("-" * 60)
    for r in results:
        q_short = r["query"][:48] + ".." if len(r["query"]) > 50 else r["query"]
        print(f"{q_short:<50} {r['precision']:>5.2f} {r['faithfulness']:>5.1f} {r['completeness']:>5.1f}")

    output_file = "eval_report.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_file}")


if __name__ == "__main__":
    main()
