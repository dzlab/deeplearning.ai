import pandas as pd
from app import assistant_chain, quiz_bank
from IPython.display import display, HTML

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

eval_system_prompt = """You are an assistant that evaluates how well the quiz assistant
    creates quizzes for a user by looking at the set of facts available to the assistant.
    Your primary concern is making sure that ONLY facts available are used. Helpful quizzes only contain facts in the
    test set"""

eval_user_message = """You are evaluating a generated quiz based on the question bank that the assistant uses to create the quiz.
  Here is the data:
    [BEGIN DATA]
    ************
    [Question Bank]: {context}
    ************
    [Quiz]: {agent_response}
    ************
    [END DATA]

## Examples of quiz questions
Subject: <subject>
   Categories: <category1>, <category2>
   Facts:
    - <fact 1>
    - <fact 2>

## Steps to make a decision
Compare the content of the submission with the question bank using the following steps

1. Review the question bank carefully. These are the only facts the quiz can reference
2. Compare the information in the quiz to the question bank.
3. Ignore differences in grammar or punctuation

Remember, the quizzes should only include information from the question bank.


## Additional rules
- Output an explanation of whether the quiz only references information in the context.
- Make the explanation brief only include a summary of your reasoning for the decsion.
- Include a clear "Yes" or "No" as the first paragraph.
- Reference facts from the quiz bank if the answer is yes

Separate the decision and the explanation. For example:

************
Decision: <Y>
************
Explanation: <Explanation>
************
"""

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


def create_eval_chain():
    eval_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", eval_system_prompt),
            ("human", eval_user_message),
        ]
    )

    return (
        eval_prompt
        | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        | StrOutputParser()
    )


def evaluate_dataset(dataset, quiz_bank, assistant, evaluator):
    eval_results = []
    for row in dataset:
        eval_result = {}
        user_input = row["input"]
        answer = assistant.invoke({"question": user_input})
        eval_response = evaluator.invoke(
            {"context": quiz_bank, "agent_response": answer}
        )

        eval_result["input"] = user_input
        eval_result["output"] = answer
        eval_result["grader_response"] = eval_response
        eval_results.append(eval_result)
    return eval_results


def report_evals():
    assistant = assistant_chain()
    model_graded_evaluator = create_eval_chain()
    eval_results = evaluate_dataset(
        dataset, quiz_bank, assistant, model_graded_evaluator
    )
    df = pd.DataFrame(eval_results)
    ## clean up new lines to be html breaks
    df_html = df.to_html().replace("\\n", "<br>")
    with open("/tmp/eval_results.html", "w") as f:
        f.write(df_html)


def main():
    report_evals()


if __name__ == "__main__":
    main()
