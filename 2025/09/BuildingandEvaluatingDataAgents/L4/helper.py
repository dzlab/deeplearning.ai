from __future__ import annotations
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportIncompatibleMethodOverride=false
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Valid config keys have changed in V2",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"WARNING! response_format is not default parameter",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"^munch$",
)

import os
import json
import re
from dotenv import load_dotenv
from snowflake.snowpark import Session
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing import Annotated, Literal, Optional, List, Dict, Any, Type
from trulens.otel.semconv.trace import SpanAttributes
from trulens.core.otel.instrument import instrument
from snowflake.core import Root
from snowflake.core.cortex.lite_agent_service import AgentRunRequest
from pydantic import BaseModel, PrivateAttr
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.schema import HumanMessage
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector
from trulens.providers.openai import OpenAI
import numpy as np
from prompts import plan_prompt, executor_prompt, agent_system_prompt



os.environ["TRULENS_OTEL_TRACING"] = "1"

# load full dotenv
load_dotenv()

# Custom State class with specific keys
class State(MessagesState):
    enabled_agents: Optional[List[str]]
    # Current plan only: mapping from step number (as string) to step definition
    plan: Optional[Dict[str, Dict[str, Any]]]
    user_query: Optional[str]
    current_step: int
    replan_flag: Optional[bool]
    last_reason: Optional[str]
    # Replan attempts tracked per step number
    replan_attempts: Optional[Dict[int, int]]
    agent_query: Optional[str]

MAX_REPLANS = 2

# Create a Snowflake session
snowflake_connection_parameters = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PAT"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
}

snowpark_session = Session.builder.configs(
    snowflake_connection_parameters
).create()

# create a python repl tool for importing in the lessons
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. You will be used to execute python code
    that generates charts. Only print the chart once.
    This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = (
        f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    )
    return (
        result_str
        + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

reasoning_llm = ChatOpenAI(
    model="o3",
    model_kwargs={"response_format": {"type": "json_object"}},
)

def planner_node(state: State) \
        -> "Command[Literal['executor']]":
    """
    Runs the planning LLM and stores the resulting plan in state.
    """
    # 1. Invoke LLM with the planner prompt
    llm_reply = reasoning_llm.invoke([plan_prompt(state)])

    # 2. Validate JSON
    try:
        content_str = llm_reply.content if isinstance(llm_reply.content, str) else str(llm_reply.content)
        parsed_plan = json.loads(content_str)
    except json.JSONDecodeError:
        raise ValueError(f"Planner returned invalid JSON:\n{llm_reply.content}")

    # 3. Store as current plan only
    replan         = state.get("replan_flag", False)
    updated_plan: Dict[str, Any] = parsed_plan

    return Command(
        update={
            "plan":         updated_plan,
            "messages":     [HumanMessage(
                                content=llm_reply.content,
                                name="replan" if replan else "initial_plan"
                             )],
            "user_query":   state.get("user_query",
                                      state["messages"][0].content),
           "current_step": 1 if not replan else state["current_step"],
           # Preserve replan flag so executor runs planned agent once before reconsidering
           "replan_flag":  state.get("replan_flag", False),
           "last_reason":  "",
           "enabled_agents": state.get("enabled_agents"),
        },
        goto="executor",
    )


# ## Create executor
# ────────────────────────────────────────────────────────────────────────
def executor_node(
    state: State,
) -> Command[Literal["web_researcher", "cortex_researcher", "chart_generator", "synthesizer", "planner"]]:

    plan: Dict[str, Any] = state.get("plan", {})
    step: int = state.get("current_step", 1)

    # 0) If we *just* replanned, run the planned agent once before reconsidering.
    if state.get("replan_flag"):
        planned_agent = plan.get(str(step), {}).get("agent")
        return Command(
            update={
                "replan_flag": False,
                "current_step": step + 1,  # advance because we executed the planned agent
            },
            goto=planned_agent,
        )

    # 1) Build prompt & call LLM
    llm_reply = reasoning_llm.invoke([executor_prompt(state)])
    try:
        content_str = llm_reply.content if isinstance(llm_reply.content, str) else str(llm_reply.content)
        parsed = json.loads(content_str)
        replan: bool = parsed["replan"]
        goto: str   = parsed["goto"]
        reason: str = parsed["reason"]
        query: str  = parsed["query"]
    except Exception as exc:
        raise ValueError(f"Invalid executor JSON:\n{llm_reply.content}") from exc

    # Upodate the state
    updates: Dict[str, Any] = {
        "messages": [HumanMessage(content=llm_reply.content, name="executor")],
        "last_reason": reason,
        "agent_query": query,
    }

    # Replan accounting
    replans: Dict[int, int] = state.get("replan_attempts", {}) or {}
    step_replans = replans.get(step, 0)

    # 2) Replan decision
    if replan:
        if step_replans < MAX_REPLANS:
            replans[step] = step_replans + 1
            updates.update({
                "replan_attempts": replans,
                "replan_flag": True,     # ensure next turn executes the planned agent once
                "current_step": step,    # stay on same step for the new plan
            })
            return Command(update=updates, goto="planner")
        else:
            # Cap hit: skip this step; let next step (or synthesizer) handle termination
            next_agent = plan.get(str(step + 1), {}).get("agent", "synthesizer")
            updates["current_step"] = step + 1
            return Command(update=updates, goto=next_agent)

    # 3) Happy path: run chosen agent; advance only if following the plan
    planned_agent = plan.get(str(step), {}).get("agent")
    updates["current_step"] = step + 1 if goto == planned_agent else step
    updates["replan_flag"] = False
    return Command(update=updates, goto=goto)

# Set semantic model file (for analyst) and search service name
SEMANTIC_MODEL_FILE = "@sales_intelligence.data.models/sales_metrics_model.yaml"

CORTEX_SEARCH_SERVICE = "sales_intelligence.data.sales_conversation_search"

# ---- Agent Setup ----
class CortexAgentArgs(BaseModel):
    query: str

class CortexAgentTool:
    name: str = "CortexAgent"
    description: str = "answers questions using sales conversations and metrics"
    args_schema: Type[CortexAgentArgs] = CortexAgentArgs

    _session: Session = PrivateAttr()
    _root: Root = PrivateAttr()
    _agent_service: Any = PrivateAttr()

    def __init__(self, session: Session):
        self._session = session
        self._root = Root(session)
        self._agent_service = self._root.cortex_agent_service

    def _build_request(self, query: str) -> AgentRunRequest:
        return AgentRunRequest.from_dict({
            "model": "claude-3-5-sonnet",
            "tools": [
                {"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}},
                {"tool_spec": {"type": "cortex_search", "name": "search1"}},
            ],
            "tool_resources": {
                "analyst1": {"semantic_model_file": SEMANTIC_MODEL_FILE},
                "search1": {
                    "name": CORTEX_SEARCH_SERVICE,
                    "max_results": 10,
                    "id_column": "conversation_id"
                }
            },
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": query}]}
            ]
        })

    def _consume_stream(self, stream):
        text, sql, citations = "", "", []
        for evt in stream.events():
            try:
                delta = (evt.data.get("delta") if isinstance(evt.data, dict)
                         else json.loads(evt.data).get("delta")
                         or json.loads(evt.data).get("data", {}).get("delta"))
            except Exception:
                continue

            if not isinstance(delta, dict):
                continue

            for item in delta.get("content", []):
                if item.get("type") == "text":
                    text += item.get("text", "")
                elif item.get("type") == "tool_results":
                    for result in item["tool_results"].get("content", []):
                        if result.get("type") != "json":
                            continue
                        j = result["json"]
                        text += j.get("text", "")
                        sql = j.get("sql", sql)
                        citations.extend({
                            "source_id": s.get("source_id"),
                            "doc_id": s.get("doc_id")
                        } for s in j.get("searchResults", []))
        return text, sql, str(citations)

    def run(self, query: str, **kwargs):
        """
        This agent will retrieve sales-related data from Snowflake using both Text2SQL and Semantic Search.
        """
        req = self._build_request(query)
        stream = self._agent_service.run(req)
        text, sql, citations = self._consume_stream(stream)

        results_str = ""
        if sql:
            try:
                # Ensure warehouse is set explicitly before running the SQL
                self._session.sql("USE WAREHOUSE SALES_INTELLIGENCE_WH").collect()

                df = self._session.sql(sql.rstrip(";")).to_pandas()
                results_str = df.to_string(index=False)
            except Exception as e:
                results_str = f"SQL execution error: {e}"

        return text, citations, sql, results_str

cortex_agent_tool = CortexAgentTool(session=snowpark_session)

from langgraph.prebuilt import create_react_agent
from helper import agent_system_prompt
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

cortex_agent = create_react_agent(llm, tools=[cortex_agent_tool.run], prompt=agent_system_prompt(f"""
        You are the Researcher. You can answer questions 
        using customer deal data along with meeting notes.
        Do not take any further action.
    """))

@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes=lambda ret, exception, *args, **kwargs: {
        SpanAttributes.RETRIEVAL.QUERY_TEXT: args[0].get("agent_query") if args[0].get("agent_query") else None,
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [
            ret.update["messages"][-1].content
        ] if hasattr(ret, "update") else "No tool call",
    },
)
def cortex_agents_research_node(
    state: State,
) -> Command[Literal["executor"]]:
    query = state.get("agent_query", state.get("user_query", ""))
    # Call the tool with the string query
    agent_response = cortex_agent.invoke({"messages":query})
    # Compose a message content string with all results new HumanMessage with the result
    new_message = HumanMessage(content=agent_response['messages'][-1].content, name="cortex_researcher")
    # Append to the message history
    goto = "executor"
    return Command(
        update={"messages": [new_message]},
        goto=goto,
    )

# ## Create Web Search Agent

tavily_tool = TavilySearch(max_results=5)

llm = ChatOpenAI(model="gpt-4o")

# Research agent and node
web_search_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=agent_system_prompt(f"""
        You are the Researcher. You can ONLY perform research by using the provided search tool (tavily_tool). 
        When you have found the necessary information, end your output.  
        Do NOT attempt to take further actions.
    """),
)

def web_research_node(
    state: State,
) -> Command[Literal["executor"]]:
    agent_query = state.get("agent_query")
    result = web_search_agent.invoke({"messages":agent_query})
    goto = "executor"
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="web_researcher"
    )
    return Command(
        update={
            # share internal message history of research agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )

# ## Create Charting Agent

# Chart generator agent and node
# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
chart_agent = create_react_agent(
    llm,
    [python_repl_tool],
    prompt=agent_system_prompt(
        "You can only generate charts. You are working with a researcher colleague. Print the chart first. Then, save the chart to a file in the current working directory and provide the path to the chart_summarizer."
    ),
)

def chart_node(state: State) -> Command[Literal["chart_summarizer"]]:
    result = chart_agent.invoke(state)
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    goto="chart_summarizer"
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )


# ## Create Chart Summary Agent

chart_summary_agent = create_react_agent(
    llm,
    tools=[],  # Add image processing tools if available/needed.
    prompt=agent_system_prompt(
        "You can only summarize the chart that was generated by the chart generator to answer the user's question. You are working with a researcher colleague and a chart generator colleague. "
        + "Your task is to generate a standalone, concise summary for the provided chart image saved at a local PATH, where the PATH should be and only be provided by your chart generator colleague. The summary should be no more than 3 sentences and should not mention the chart itself."
    ),
)

def chart_summary_node(
    state: State,
) -> Command[Literal[END]]:
    result = chart_summary_agent.invoke(state)
    print(f"Chart summarizer answer: {result['messages'][-1].content}")
    # Ensure the summary message is attributed to chart_summarizer for downstream use
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_summarizer"
    )
    # Send to the end node
    goto = END
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
            "final_answer": result["messages"][-1].content,
        },
        goto=goto,
    )


# ## Create a Synthesizer Agent
def synthesizer_node(state: State) -> Command[Literal[END]]:
    """
    Creates a concise, human‑readable summary of the entire interaction,
    **purely in prose**.

    It ignores structured tables or chart IDs and instead rewrites the
    relevant agent messages (research results, chart commentary, etc.)
    into a short final answer.
    """
    # Gather informative messages for final synthesis
    relevant_msgs = [
        m.content for m in state.get("messages", [])
        if getattr(m, "name", None) in ("web_researcher", "cortex_researcher", "chart_generator", "chart_summarizer")
    ]

    user_question = state.get("user_query", state.get("messages", [{}])[0].content if state.get("messages") else "")

    synthesis_instructions = (
            "You are the Synthesizer. Use the context below to directly answer the user's question. " # UPDATED THIS LINE
            "Perform any lightweight calculations, comparisons, or inferences required. " # ADDED THIS LINE
            "Do not invent facts not supported by the context. If data is missing, say what's missing and, if helpful, " # UPDATED THIS LINE
            "offer a clearly labeled best-effort estimate with assumptions.\n\n" # ADDED THIS LINE
            "Produce a concise response that fully answers the question, with the following guidance:\n" # UPDATED THIS LINE
            "- Start with the direct answer (one short paragraph or a tight bullet list).\n"
            "- Include key figures from any 'Results:' tables (e.g., totals, top items).\n"
            "- If any message contains citations, include them as a brief 'Citations: [...]' line.\n"
            "- Keep the output crisp; avoid meta commentary or tool instructions."
        )

    summary_prompt = [
        HumanMessage(content=(
            f"User question: {user_question}\n\n"
            f"{synthesis_instructions}\n\n"
            f"Context:\n\n" + "\n\n---\n\n".join(relevant_msgs)
        ))
    ]
    llm_reply = llm.invoke(summary_prompt)

    reply_content = llm_reply.content
    if isinstance(reply_content, list):
        reply_text = "".join([c if isinstance(c, str) else str(c) for c in reply_content])
    else:
        reply_text = str(reply_content)
    answer = reply_text.strip()
    print(f"Synthesizer answer: {answer}")

    return Command(
        update={
            "final_answer": answer,
            "messages": [HumanMessage(content=answer, name="synthesizer")],
        },
        goto=END,           # hand off to the END node
    )

# Evaluations

# Use GPT-4o for RAG Triad Evaluations
provider = OpenAI(model_engine="gpt-4o")

# Define a groundedness feedback function
f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    )
    .on({
            "source": Selector(
                span_type=SpanAttributes.SpanType.RETRIEVAL,
                span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
                collect_list=True
            )
        }
    )
    .on_output()
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input()
    .on_output()
)

# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on({
            "question": Selector(
                span_type=SpanAttributes.SpanType.RETRIEVAL,
                span_attribute=SpanAttributes.RETRIEVAL.QUERY_TEXT,
            )
        }
    )
    .on({
            "context": Selector(
                span_type=SpanAttributes.SpanType.RETRIEVAL,
                span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
                collect_list=False
            )
        }
    )
    .aggregate(np.mean)
)

# Use GPT-4.1 for Goal-Plan-Act
gpa_eval_provider = OpenAI(model_engine="gpt-4.1")

# Goal-Plan-Act: Logical consistency of trace
f_logical_consistency = Feedback(
    gpa_eval_provider.logical_consistency_with_cot_reasons,
    name="Logical Consistency",
).on({
    "trace": Selector(trace_level=True),
})

# Goal-Plan-Act: Execution efficiency of trace
f_execution_efficiency = Feedback(
    gpa_eval_provider.execution_efficiency_with_cot_reasons,
    name="Execution Efficiency",
).on({
    "trace": Selector(trace_level=True),
})

# Goal-Plan-Act: Plan adherence
f_plan_adherence = Feedback(
    gpa_eval_provider.plan_adherence_with_cot_reasons,
    name="Plan Adherence",
).on({
    "trace": Selector(trace_level=True),
})

# Goal-Plan-Act: Plan quality
f_plan_quality = Feedback(
    gpa_eval_provider.plan_quality_with_cot_reasons,
    name="Plan Quality",
).on({
    "trace": Selector(trace_level=True),
})

from IPython.display import HTML, display

def display_eval_reason(text, width=800):
    # Strip any trailing "Score: X" from the end of the text
    raw_text = str(text).rstrip()
    cleaned_text = re.sub(r"\s*Score:\s*-?\d+(?:\.\d+)?\s*$", "", raw_text, flags=re.IGNORECASE)
    # Convert newlines to HTML line breaks, then wrap
    html_text = cleaned_text.replace('\n', '<br><br>')
    display(HTML(f'<div style="font-size: 15px; word-wrap: break-word; width: {width}px;">{html_text}</div>'))