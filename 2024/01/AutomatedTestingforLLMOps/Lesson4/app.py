from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


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

delimiter = "####"

system_message = f"""
Follow these steps to generate a customized quiz for the user.
The question will be delimited with four hashtags i.e {delimiter}

Step 1:{delimiter} First identify the category user is asking about from the following list:
* Geography
* Science
* Art

Step 2:{delimiter} Determine the subjects to generate questions about. The list of topics are below:

{quiz_bank}

Pick up to two subjects that fit the user's category.

Step 3:{delimiter} Generate a quiz for the user. Based on the selected subjects generate 3 questions for the user using the facts about the subject.
* Only ask questions about facts that are in the list of topics
* Do not include additional facts

Use the following format:
Question 1:{delimiter} <question 1>

Question 2:{delimiter} <question 2>

Question 3:{delimiter} <question 3>

"""


def assistant_chain():
    human_template = "{question}"

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", human_template),
        ]
    )
    return (
        chat_prompt
        | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        | StrOutputParser()
    )
