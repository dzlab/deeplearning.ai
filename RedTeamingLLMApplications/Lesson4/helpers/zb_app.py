"""ZephyrBank chatbot: a demo LLM app used in the course."""

import time
from typing import List

from llama_index import (
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.llms.types import ChatMessage
from llama_index.llms import OpenAI
from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever
from pathlib import Path
from llama_index.chat_engine.condense_question import (
    CondenseQuestionChatEngine,
)

STORAGE_DIR = Path(__file__).parent / "data" / "zb_vstore"
OPENAI_MODEL = "gpt-3.5-turbo-0613"

QA_PROMPT = """You are an expert Q&A system for ZephyrBank, a fintech company specializing in banking services for business owners.

Always answer the user question. You are given some context information to help you in answering.
Avoid statements like 'Based on the context', 'The context information', 'The context does not contain', 'The context does not mention', 'in the given context', or anything similar.

### Context:
{context_str}

### Query:
{query_str}

### Answer:
"""

REFINE_PROMPT = """The original query is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer with some more context below.
------------
{context_msg}
------------
Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.
Refined Answer: """


CONDENSE_PROMPT = """Given a conversation (between Human and Assistant) and a follow up message from Human, rewrite the message to be a standalone question that captures all relevant context from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>"""


class RAGQueryEngine(CustomQueryEngine):
    retriever: BaseRetriever
    llm: OpenAI
    refine_answer: bool

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        context_str = "\n".join([n.node.get_content() for n in nodes if n.score > 0.77])

        # for node in nodes:
        #     print(f"[{node.score}] {node.get_content()}")

        response = self.llm.complete(
            PromptTemplate(QA_PROMPT).format(
                context_str=context_str, query_str=query_str
            ),
        )

        if context_str or self.refine_answer:
            response = self.llm.complete(
                PromptTemplate(REFINE_PROMPT).format(
                    query_str=query_str,
                    existing_answer=str(response),
                    context_msg=context_str,
                ),
            )

        return str(response)


def get_retriever():
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    vs = load_index_from_storage(storage_context)

    return vs.as_retriever()


def make_app():
    llm = OpenAI(temperature=0.5, model=OPENAI_MODEL)
    retriever = get_retriever()
    query_engine = RAGQueryEngine(retriever=retriever, llm=llm)

    def model_fn(query: str):
        return query_engine.query(query).response

    return model_fn


class CustomChatEngine(CondenseQuestionChatEngine):
    def _condense_question(
        self, chat_history: List[ChatMessage], last_message: str
    ) -> str:
        if len(chat_history) == 0:
            return last_message

        return super()._condense_question(chat_history, last_message)


class ZephyrApp:
    def __init__(self, version="v1"):
        self._version = version.lower()
        self._llm = OpenAI(temperature=0.1, model=OPENAI_MODEL)
        retriever = get_retriever()
        self._query_engine = RAGQueryEngine(
            retriever=retriever, llm=self._llm, refine_answer=self._version == "v2"
        )
        self._chat_engine = CustomChatEngine.from_defaults(
            condense_question_prompt=PromptTemplate(CONDENSE_PROMPT),
            query_engine=self._query_engine,
            llm=self._llm,
        )

    def chat(self, message: str):
        if len(message) > 8_000:
            time.sleep(5)
            return "API ERROR: Request Timeout"

        return self._chat_engine.chat(message).response

    def reset(self):
        self._chat_engine.reset()


class Conversation:
    def __init__(self, model_fn):
        self.model_fn = model_fn
        self.messages = []

    def message(self, message):
        self.messages.append({"role": "user", "content": message})
        answer = self.model_fn(self.messages)
        self.messages.append({"role": "assistant", "content": answer})
        return answer
