from app import assistant_chain, quiz_bank

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


def create_eval_chain(context, agent_response):
    eval_system_prompt = """You are an assistant that evaluates how well the quiz assistant
    creates quizzes for a user by looking at the set of facts available to the assistant.
    Your primary concern is making sure that ONLY facts available are used. Helpful quizzes only contain facts in the
    test set"""

    eval_user_message = f"""You are evaluating a generated quiz based on the context that the assistant uses to create the quiz.
  Here is the data:
    [BEGIN DATA]
    ************
    [Question Bank]: {context}
    ************
    [Quiz]: {agent_response}
    ************
    [END DATA]

Compare the content of the submission with the question bank using the following steps

1. Review the question bank carefully. These are the only facts the quiz can reference
2. Compare the quiz to the question bank.
3. Ignore differences in grammar or punctuation
4. If a fact is in the quiz, but not in the question bank the quiz if bad.

Remember, the quizzes need to only include facts the assistant is aware of. It is dangerous to allow made up facts.

Output Y if the quiz only contains facts from the question bank, output N if it contains facts that are not in the question bank.
"""
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


def test_model_graded_eval_hallucination():
    assistant = assistant_chain()
    quiz_request = "Write me a quiz about books."
    result = assistant.invoke({"question": quiz_request})
    print(result)
    eval_agent = create_eval_chain(quiz_bank, result)
    eval_response = eval_agent.invoke({})
    print(eval_response)
    # Our test asks about a subject not in the context, so the agent should answer N
    assert eval_response == "N"
