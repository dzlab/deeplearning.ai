from app import assistant_chain

# In a real application you would load your dataset from a file or logging tool.
# Here we have a mix of examples with slightly different phrasing that our quiz application can support
# and things we don't support.
dataset = [
    {
        "input": "I'm trying to learn about science, can you give me a quiz to test my knowledge",
        "response": "science",
        "subjects": ["davinci", "telescope", "physics", "curie"],
    },
    {
        "input": "I'm an geography expert, give a quiz to prove it?",
        "response": "geography",
        "subjects": ["paris", "france", "louvre"],
    },
    {
        "input": "Quiz me about Italy",
        "response": "geography",
        "subjects": ["rome", "alps", "sicily"],
    },
]


def test_on_dataset():
    assistant = assistant_chain()
    for row in dataset:
        user_input = row["input"]
        expected_category = row["response"]
        expected_subjects = row.get("subjects", None)
        answer = assistant.invoke({"question": user_input})
        assert (
            expected_category.lower() in answer.lower()
        ), f"expected: {expected_category}, got {answer}"
        if expected_subjects:
            assert any(
                subject.lower() in answer.lower() for subject in expected_subjects
            ), f"Expected the assistant questions to include '{expected_subjects}', but got {answer}"
