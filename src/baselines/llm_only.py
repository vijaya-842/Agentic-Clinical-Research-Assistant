import json
import sys
from pathlib import Path


# Find the project root directory
project_root = Path(__file__).resolve().parents[2]

# Allow Python to import modules from the project root
sys.path.append(str(project_root))

from src.llm.ollama_client import generate_response


SYSTEM_PROMPT = """
You are a biomedical research assistant.

Answer the user's question based only on your existing knowledge.

Do not claim that you searched a database or retrieved scientific documents.
Do not invent citations, studies, clinical trials, or sources.

If you are uncertain about something, clearly state that uncertainty.
"""


def load_questions() -> list:
    """
    Load evaluation questions from eval/questions.json.
    """

    questions_path = project_root / "eval" / "questions.json"

    with open(questions_path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    return questions


def answer_question(question: str) -> str:
    """
    Generate an answer using only the LLM.

    No retrieval, documents, vector database,
    or external tools are used.
    """

    return generate_response(
        prompt=question,
        system_prompt=SYSTEM_PROMPT,
    )


def run_baseline():
    """
    Run all evaluation questions through the LLM-only baseline.
    """

    questions = load_questions()

    results = []

    for item in questions:

        question_id = item["id"]
        question = item["question"]
        category = item["category"]

        print(f"\n{'=' * 60}")
        print(f"Question ID: {question_id}")
        print(f"Category: {category}")
        print(f"Question: {question}")

        answer = answer_question(question)

        print("\nLLM-Only Answer:")
        print(answer)

        results.append(
            {
                "id": question_id,
                "question": question,
                "category": category,
                "answer": answer,
            }
        )

    return results

def save_results(results: list) -> None:
    """
    Save LLM-only baseline results as a JSON file.
    """

    output_path = project_root / "eval" / "llm_only_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":

    results = run_baseline()

    save_results(results)

    print(f"\n{'=' * 60}")
    print(f"Completed {len(results)} LLM-only baseline questions.")