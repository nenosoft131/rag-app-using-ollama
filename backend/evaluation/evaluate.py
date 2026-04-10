from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


def run_evaluation():

    results = []

    for row in dataset:
        output = engine.ask_with_context(row["question"])

        results.append(
            {
                "question": row["question"],
                "answer": output["answer"],
                "contexts": output["contexts"],
                "ground_truth": row["answer"],
            }
        )

    from datasets import Dataset

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

    print(scores)


if __name__ == "__main__":
    run_evaluation()
