from typing import Dict, Any, List
from langchain.schema import HumanMessage  # type: ignore[import-not-found]
import json
from typing import Optional
from langgraph.graph import MessagesState
from langgraph.types import Command
from typing import Literal, Optional, List, Dict, Any, Type

MAX_REPLANS = 2

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

def get_agent_descriptions() -> Dict[str, Dict[str, Any]]:
    """
    Return structured agent descriptions with capabilities and guidelines.
    Edit this function to change how the planner/executor reason about agents.
    """
    return {
        "web_researcher": {
            "name": "Web Researcher",
            "capability": "Fetch public data via Tavily web search",
            "use_when": "Public information, news, current events, or external facts are needed",
            "limitations": "Cannot access private/internal company data",
            "output_format": "Raw research data and findings from public sources",
        },
        "cortex_researcher": {
            "name": "Cortex Researcher",
            "capability": "Query private/company data in Snowflake, including structured deal records (company name, deal value, sales rep, close date, deal status, product line) and unstructured sales meeting notes, via Snowflake Cortex Agents.",
            "use_when": "Internal documents, company databases, or private data access is required",
            "limitations": "Cannot access public web data",
            "output_format": "For structured requests, return the exact fields and include SQL when applicable; for unstructured, return concise relevant excerpts with citations.",
        },
        "chart_generator": {
            "name": "Chart Generator",
            "capability": "Build visualizations from structured data",
            "use_when": "User explicitly requests charts, graphs, plots, visualizations (keywords: chart, graph, plot, visualise, bar-chart, line-chart, histogram, etc.)",
            "limitations": "Requires structured data input from previous steps",
            "output_format": "Visual charts and graphs",
            "position_requirement": "Must be used as final step after data gathering is complete",
        },
        "chart_summarizer": {
            "name": "Chart Summarizer",
            "capability": "Summarize and explain chart visualizations",
            "use_when": "After chart_generator has created a visualization",
            "limitations": "Requires a chart as input",
            "output_format": "Written summary and analysis of chart content",
        },
        "synthesizer": {
            "name": "Synthesizer",
            "capability": "Write comprehensive prose summaries of findings",
            "use_when": "Final step when no visualization is requested - combines all previous research",
            "limitations": "Requires research data from previous steps",
            "output_format": "Coherent written summary incorporating all findings",
            "position_requirement": "Should be used as final step when no chart is needed",
        },
    }

def _get_enabled_agents(state: State | None = None) -> List[str]:
    """Return enabled agents; if absent, use baseline/default.

    Supports both dict-style and attribute-style state objects.
    """
    baseline = ["web_researcher", "chart_generator", "chart_summarizer", "synthesizer"]
    if not state:
        return baseline
    val = state.get("enabled_agents") if hasattr(state, "get") else getattr(state, "enabled_agents", None)
    
    if isinstance(val, list) and val:
        allowed = {"web_researcher", "cortex_researcher", "chart_generator", "chart_summarizer", "synthesizer"}
        filtered = [a for a in val if a in allowed]
        return filtered
    return baseline

def format_agent_list_for_planning(state: State | None = None) -> str:
    """
    Format agent descriptions for the planning prompt.
    """
    descriptions = get_agent_descriptions()
    enabled_list = _get_enabled_agents(state)
    agent_list = []
    
    for agent_key, details in descriptions.items():
        if agent_key not in enabled_list:
            continue
        agent_list.append(f"  • `{agent_key}` – {details['capability']}")
    
    return "\n".join(agent_list)

def format_agent_guidelines_for_planning(state: State | None = None) -> str:
    """
    Format agent usage guidelines for the planning prompt.
    """
    descriptions = get_agent_descriptions()
    enabled = set(_get_enabled_agents(state))
    guidelines = []
    
    # Cortex vs Web researcher (only include guidance for enabled agents)
    if "cortex_researcher" in enabled:
        guidelines.append(f"- Use `cortex_researcher` when {descriptions['cortex_researcher']['use_when'].lower()}.")
    if "web_researcher" in enabled:
        guidelines.append(f"- Use `web_researcher` for {descriptions['web_researcher']['use_when'].lower()}.")
    
    # Chart generator specific rules
    if "chart_generator" in enabled:
        chart_desc = descriptions['chart_generator']
        cs_hint = " A `chart_summarizer` should be used to summarize the chart." if "chart_summarizer" in enabled else ""
        guidelines.append(f"- **Include `chart_generator` _only_ if {chart_desc['use_when'].lower()}**. If included, `chart_generator` must be {chart_desc['position_requirement'].lower()}. Visualizations should include all of the data from the previous steps that is reasonable for the chart type.{cs_hint}")
    
    # Synthesizer default
    if "synthesizer" in enabled:
        synth_desc = descriptions['synthesizer'] 
        guidelines.append(f"  – Otherwise use `synthesizer` as {synth_desc['position_requirement'].lower()}, and be sure to include all of the data from the previous steps.")
    
    return "\n".join(guidelines)

def format_agent_guidelines_for_executor(state: State | None = None) -> str:
    """
    Format agent usage guidelines for the executor prompt.
    """
    descriptions = get_agent_descriptions()
    enabled = _get_enabled_agents(state)
    guidelines = []
    
    if "web_researcher" in enabled:
        web_desc = descriptions['web_researcher']
        guidelines.append(f"- Use `\"web_researcher\"` when {web_desc['use_when'].lower()}.")
    if "cortex_researcher" in enabled:
        cortex_desc = descriptions['cortex_researcher']
        guidelines.append(f"- Use `\"cortex_researcher\"` for {cortex_desc['use_when'].lower()}.")
    
    return "\n".join(guidelines)

def plan_prompt(state: State) -> HumanMessage:
    """
    Build the prompt that instructs the LLM to return a high‑level plan.
    """
    replan_flag   = state.get("replan_flag", False)
    user_query    = state.get("user_query", state["messages"][0].content)
    prior_plan    = state.get("plan") or {}
    replan_reason = state.get("last_reason", "")
    
    # Get agent descriptions dynamically
    
    agent_list = format_agent_list_for_planning(state)
    agent_guidelines = format_agent_guidelines_for_planning(state)

    enabled_list = _get_enabled_agents(state)

    # Build planner agent enum based on enabled agents
    enabled_for_planner = [
        a for a in enabled_list
        if a in ("web_researcher", "cortex_researcher", "chart_generator", "synthesizer")
    ]
    planner_agent_enum = " | ".join(enabled_for_planner) or "web_researcher | chart_generator | synthesizer"

    prompt = f"""
        You are the **Planner** in a multi‑agent system.  Break the user's request
        into a sequence of numbered steps (1, 2, 3, …).  **There is no hard limit on
        step count** as long as the plan is concise and each step has a clear goal.

        You may decompose the user's query into sub-queries, each of which is a
        separate step.  Break the query into the smallest possible sub-queries
        so that each sub-query is answerable with a single data source.
        For example, if the user's query is "What were the key
        action items in the last quarter, and what was a recent news story for 
        each of them?", you may break it into steps:

        1. Fetch the key action items in the last quarter.
        2. Fetch a recent news story for the first action item.
        3. Fetch a recent news story for the second action item.
        4. Fetch a recent news story for the last action item

        Here is a list of available agents you can call upon to execute the tasks in your plan. You may call only one agent per step.

        {agent_list}

        Return **ONLY** valid JSON (no markdown, no explanations) in this form:

        {{
        "1": {{
            "agent": "{planner_agent_enum}",
            "action": "string",
        }},
        "2": {{ ... }},
        "3": {{ ... }}
        }}

        Guidelines:
        {agent_guidelines}
        """

    if replan_flag:
        prompt += f"""
        The current plan needs revision because: {replan_reason}

        Current plan:
        {json.dumps(prior_plan, indent=2)}

        When replanning:
        - Focus on UNBLOCKING the workflow rather than perfecting it.
        - Only modify steps that are truly preventing progress.
        - Prefer simpler, more achievable alternatives over complex rewrites.
        """

    else:
        prompt += "\nGenerate a new plan from scratch."

    prompt += f'\nUser query: "{user_query}"'
    
    return HumanMessage(content=prompt)

def executor_prompt(state: State) -> HumanMessage:
    """
    Build the single‑turn JSON prompt that drives the executor LLM.
    """
    step = int(state.get("current_step", 0))
    latest_plan: Dict[str, Any] = state.get("plan") or {}
    plan_block: Dict[str, Any] = latest_plan.get(str(step), {})
    max_replans    = MAX_REPLANS
    attempts       = (state.get("replan_attempts", {}) or {}).get(step, 0)
    
    # Get agent guidelines dynamically
    executor_guidelines = format_agent_guidelines_for_executor(state)
    plan_agent = plan_block.get("agent", "web_researcher")

    messages_tail = (state.get("messages") or [])[-4:]

    executor_prompt = f"""
        You are the **executor** in a multi‑agent system with these agents:
        `{ '`, `'.join(sorted(set([a for a in _get_enabled_agents(state) if a in ['web_researcher','cortex_researcher','chart_generator','chart_summarizer','synthesizer']] + ['planner']))) }`.

        **Tasks**
        1. Decide if the current plan needs revision.  → `"replan_flag": true|false`
        2. Decide which agent to run next.             → `"goto": "<agent_name>"`
        3. Give one‑sentence justification.            → `"reason": "<text>"`
        4. Write the exact question that the chosen agent should answer
                                                    → "query": "<text>"

        **Guidelines**
        {executor_guidelines}
        - After **{MAX_REPLANS}** failed replans for the same step, move on.
        - If you *just replanned* (replan_flag is true) let the assigned agent try before
        requesting another replan.

        Respond **only** with valid JSON (no additional text):

        {{
        "replan": <true|false>,
        "goto": "<{ '|'.join([a for a in _get_enabled_agents(state) if a in ['web_researcher','cortex_researcher','chart_generator','chart_summarizer','synthesizer']] + ['planner']) }>",
        "reason": "<1 sentence>",
        "query": "<text>"
        }}

        **PRIORITIZE FORWARD PROGRESS:** Only replan if the current step is completely blocked.
        1. If any reasonable data was obtained that addresses the step's core goal, set `"replan": false` and proceed.
        2. Set `"replan": true` **only if** ALL of these conditions are met:
        • The step has produced zero useful information
        • The missing information cannot be approximated or obtained by remaining steps
        • `attempts < {max_replans}`
        3. When `attempts == {max_replans}`, always move forward (`"replan": false`).

        ### Decide `"goto"`
        - If `"replan": true` → `"goto": "planner"`.
        - If current step has made reasonable progress → move to next step's agent.
        - Otherwise execute the current step's assigned agent (`{plan_agent}`).

        ### Build `"query"`
        Write a clear, standalone instruction for the chosen agent. If the chosen agent 
        is `web_researcher` or `cortex_researcher`, the query should be a standalone question, 
        written in plain english, and answerable by the agent.

        Ensure that the query uses consistent language as the user's query.

        Context you can rely on
        - User query ..............: {state.get("user_query")}
        - Current step index ......: {step}
        - Current plan step .......: {plan_block}
        - Just‑replanned flag .....: {state.get("replan_flag")}
        - Previous messages .......: {messages_tail}

        Respond **only** with JSON, no extra text.
        """

    return HumanMessage(
        content=executor_prompt
    )

def agent_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )