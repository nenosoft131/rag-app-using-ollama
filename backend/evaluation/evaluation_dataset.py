from datasets import Dataset


def generate_eval_dataset_from_docs(documents):
    """
    documents = list of LangChain Document objects
    """

    questions = []
    answers = []
    contexts = []

    for doc in documents:
        text = doc.page_content

        # Simple heuristic generation (can be improved with LLM later)
        if len(text) < 100:
            continue

        question = f"What is described in this text: {text[:50]}?"
        answer = text[:200]

        questions.append(question)
        answers.append(answer)
        contexts.append([text])

    return Dataset.from_dict(
        {"question": questions, "ground_truth": answers, "contexts": contexts}
    )
