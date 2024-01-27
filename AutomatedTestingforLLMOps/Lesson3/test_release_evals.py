from app import assistant_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
import pytest


def create_eval_chain(
    agent_response,
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    output_parser=StrOutputParser(),
):
    delimiter = "####"
    eval_system_prompt = f"""You are an assistant that evaluates whether or not an assistant is producing valid quizzes.
  The assistant should be producing output in the format of Question N:{delimiter} <question N>?"""

    eval_user_message = f"""You are evaluating a generated quiz based on the context that the assistant uses to create the quiz.
  Here is the data:
    [BEGIN DATA]
    ************
    [Response]: {agent_response}
    ************
    [END DATA]

Read the response carefully and determine if it looks like a quiz or test. Do not evaluate if the information is correct
only evaluate if the data is in the expected format.

Output Y if the response is a quiz, output N if the response does not look like a quiz.
"""
    eval_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", eval_system_prompt),
            ("human", eval_user_message),
        ]
    )

    return eval_prompt | llm | output_parser


@pytest.fixture
def known_bad_result():
    return "There are lots of interesting facts. Tell me more about what you'd like to know"


@pytest.fixture
def quiz_request():
    return "Give me a quiz about Geography"


def test_model_graded_eval(quiz_request):
    assistant = assistant_chain()
    result = assistant.invoke({"question": quiz_request})
    print(result)
    eval_agent = create_eval_chain(result)
    eval_response = eval_agent.invoke({})
    assert eval_response == "Y"


def test_model_graded_eval_should_fail(known_bad_result):
    print(known_bad_result)
    eval_agent = create_eval_chain(known_bad_result)
    eval_response = eval_agent.invoke({})
    assert (
        eval_response == "Y"
    ), f"expected failure, asserted the response should be 'Y', \
    got back '{eval_response}'"
