from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
import random


# -----------------------------
# 1. Base Dataset (Seed Data)
# -----------------------------
def load_base_dataset():
    return [
        {
            "question": "What is Python?",
            "answer": "Python is a programming language.",
            "contexts": [
                "Python is a high-level programming language used for general-purpose programming."
            ],
        },
        {
            "question": "Who created ChatGPT?",
            "answer": "OpenAI",
            "contexts": ["ChatGPT is an AI model developed by OpenAI."],
        },
    ]


# -----------------------------
# 2. Dataset Expansion
# -----------------------------
def expand_dataset(dataset):
    paraphrase_templates = [
        "Can you explain {}?",
        "What is meant by {}?",
        "Give details about {}.",
        "Provide an explanation of {}.",
    ]

    expanded = []

    for row in dataset:
        # Keep original
        expanded.append(row)

        # Add paraphrases
        for template in paraphrase_templates:
            expanded.append(
                {
                    "question": template.format(row["question"]),
                    "answer": row["answer"],
                    "contexts": row.get("contexts", []),
                }
            )

    # Add multi-hop example
    expanded.append(
        {
            "question": "Who is the CEO of the company that created ChatGPT?",
            "answer": "Sam Altman",
            "contexts": [
                "ChatGPT is developed by OpenAI.",
                "Sam Altman is the CEO of OpenAI.",
            ],
        }
    )

    # Add unanswerable example
    expanded.append(
        {
            "question": "What is the salary of OpenAI's CEO?",
            "answer": "Not available",
            "contexts": [],
        }
    )

    # Shuffle dataset
    random.shuffle(expanded)

    return expanded


# -----------------------------
# 3. Dummy RAG Engine (Replace This)
# -----------------------------
class DummyEngine:
    def ask_with_context(self, question):
        """
        Replace this with your actual RAG pipeline.
        Must return:
        {
            "answer": str,
            "contexts": List[str]
        }
        """
        return {
            "answer": "This is a dummy answer.",
            "contexts": ["This is a dummy retrieved context."],
        }


# -----------------------------
# 4. Run Evaluation
# -----------------------------
def run_evaluation():
    # Load + expand dataset
    base_dataset = load_base_dataset()
    dataset = expand_dataset(base_dataset)

    engine = DummyEngine()

    results = []

    for row in dataset:
        output = engine.ask_with_context(row["question"])

        results.append(
            {
                "question": row["question"],
                "answer": output["answer"],
                "contexts": output["contexts"],
                "ground_truth": row["answer"],
                "ground_truth_contexts": row.get("contexts", []),
            }
        )

    eval_dataset = Dataset.from_list(results)

    scores = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    print("\n=== Evaluation Scores ===")
    print(scores)


# -----------------------------
# 5. Entry Point
# -----------------------------
if __name__ == "__main__":
    run_evaluation()
