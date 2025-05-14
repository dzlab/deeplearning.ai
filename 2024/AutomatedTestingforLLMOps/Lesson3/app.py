from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

delimiter = "####"


def read_file_into_string(file_path):
    try:
        with open(file_path, "r") as file:
            file_content = file.read()
            return file_content
    except FileNotFoundError:
        print(f"The file at '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


quiz_bank = read_file_into_string("quiz_bank.txt")

system_message = f"""
Follow these steps to generate a customized quiz for the user.
The question will be delimited with four hashtags i.e {delimiter}

The user will provide a category that they want to create a quiz for. Any questions included in the quiz
should only refer to the category.

Step 1:{delimiter} First identify the category user is asking about from the following list:
* Geography
* Science
* Art

Step 2:{delimiter} Determine the subjects to generate questions about. The list of topics are in the quiz bank below:

#### Start Quiz Bank
{quiz_bank}

#### End Quiz Bank

Pick up to two subjects that fit the user's category. 

Step 3:{delimiter} Generate a quiz for the user. Based on the selected subjects generate 3 questions for the user using the facts about the subject.

* Only include questions for subjects that are in the quiz bank.

Use the following format for the quiz:
Question 1:{delimiter} <question 1>

Question 2:{delimiter} <question 2>

Question 3:{delimiter} <question 3>

Additional rules:
- Only include questions from information in the quiz bank. Students only know answers to questions from the quiz bank, do not ask them about other topics.
- Only use explicit string matches for the category name, if the category is not an exact match for Geography, Science, or Art answer that you do not have information on the subject.
- If the user asks a question about a subject you do not have information about in the quiz bank, answer "I'm sorry I do not have information about that".
"""

"""
  Helper functions for writing the test cases
"""


def assistant_chain(
    system_message=system_message,
    human_template="{question}",
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    output_parser=StrOutputParser(),
):
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", human_template),
        ]
    )
    return chat_prompt | llm | output_parser
