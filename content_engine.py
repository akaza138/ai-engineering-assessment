"""
Phase 2: The Autonomous Content Engine (LangGraph)
===================================================
A LangGraph state machine with three nodes:
  Node 1 – Decide Search  : LLM picks today's topic & formats a search query
  Node 2 – Web Search     : Calls mock_searxng_search tool for real-world context
  Node 3 – Draft Post     : LLM drafts a 280-char opinionated post in persona

Output is always a strict JSON object:
  {"bot_id": "...", "topic": "...", "post_content": "..."}
"""

import os
import json
import re
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

load_dotenv()


# ─────────────────────────────────────────────
# LLM Factory  (xAI → Anthropic → Groq → OpenAI → Ollama fallback)
# ─────────────────────────────────────────────

def get_llm(temperature: float = 0.7):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "xai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="grok-3-mini",
            temperature=temperature,
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            temperature=temperature,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama3-8b-8192",
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        # Ollama (local)
        from langchain_ollama import ChatOllama
        return ChatOllama(model="gemma3:4b", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))


# ─────────────────────────────────────────────
# Mock Search Tool
# ─────────────────────────────────────────────

MOCK_NEWS_DB: dict = {
    "crypto":     "Bitcoin hits new all-time high amid regulatory ETF approvals. Ethereum surges 40%.",
    "bitcoin":    "Bitcoin hits new all-time high amid regulatory ETF approvals. Ethereum surges 40%.",
    "ai":         "OpenAI releases GPT-5; Anthropic claims Claude 4 beats it on reasoning benchmarks.",
    "openai":     "OpenAI releases GPT-5; Anthropic claims Claude 4 beats it on reasoning benchmarks.",
    "elon":       "Elon Musk's xAI launches Grok-3, claims superiority over all frontier models.",
    "space":      "SpaceX Starship completes first fully successful orbit; Musk says Mars by 2027.",
    "tech":       "Big Tech layoffs hit 50,000 in Q1 2026 as AI automation accelerates.",
    "regulation": "EU AI Act enforcement begins; dozens of startups face compliance deadlines.",
    "privacy":    "Meta fined €2B for GDPR violations; whistleblower exposes secret data brokering.",
    "capitalism": "Amazon warehouse workers strike across 12 countries citing AI speed-up quotas.",
    "billionaire":"Bezos surpasses Musk as world's richest person after Blue Origin IPO.",
    "fed":        "Fed holds rates at 4.5%; Powell signals two cuts possible before year-end.",
    "interest":   "10-year Treasury yields hit 5.2% as inflation ticks back up to 3.8%.",
    "market":     "S&P 500 hits all-time high of 6,200; tech sector leads with 18% YTD gains.",
    "stock":      "Nvidia stock up 300% YTD as AI chip demand shows no signs of slowing.",
    "trading":    "Quant funds outperform human traders by 23% in volatile Q1 2026 markets.",
    "ev":         "Tesla reports 35% drop in deliveries as EV market faces battery supply crunch.",
    "climate":    "Global CO2 levels hit record high; IPCC warns of irreversible tipping points.",
    "default":    "Global tech stocks rally on strong earnings; AI sector leads gains.",
}


@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a SearXNG web search by returning hardcoded recent
    news headlines matched on keywords in the query.

    Args:
        query: A natural language or keyword search query.

    Returns:
        A string containing recent news headlines relevant to the query.
    """
    query_lower = query.lower()
    for keyword, headline in MOCK_NEWS_DB.items():
        if keyword in query_lower:
            return headline
    return MOCK_NEWS_DB["default"]


# ─────────────────────────────────────────────
# LangGraph State
# ─────────────────────────────────────────────

class PostState(TypedDict):
    bot_id:        str
    persona:       str
    search_query:  str
    search_result: str
    topic:         str
    post_content:  str
    final_json:    dict


# ─────────────────────────────────────────────
# Structured Output Schema
# ─────────────────────────────────────────────

class BotPost(BaseModel):
    bot_id:       str  = Field(description="The bot's ID, e.g. bot_a")
    topic:        str  = Field(description="The topic the bot chose to post about")
    post_content: str  = Field(description="The final post text, max 280 characters")


# ─────────────────────────────────────────────
# Node Functions
# ─────────────────────────────────────────────

def node_decide_search(state: PostState) -> PostState:
    """
    Node 1 – Decide Search
    The LLM looks at the bot's persona and decides what topic to post about,
    then formats a concise search query.
    """
    print(f"\n[Node 1] Deciding search topic for {state['bot_id']} …")
    llm = get_llm(temperature=0.8)

    prompt = f"""You are {state['bot_id']}, a social media bot with this persona:
"{state['persona']}"

Today you want to make an opinionated post. Based ONLY on your persona, decide:
1. What topic is most relevant to you right now?
2. What is a short 3-5 word search query to find recent news on that topic?

Reply with ONLY a JSON object like:
{{"topic": "...", "search_query": "..."}}
No markdown, no explanation."""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Strip markdown fences if the model adds them
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        parsed = json.loads(raw)
        state["topic"]        = parsed["topic"]
        state["search_query"] = parsed["search_query"]
    except Exception:
        # Fallback if LLM output is malformed
        state["topic"]        = "technology"
        state["search_query"] = "latest tech news 2026"

    print(f"  → Topic: {state['topic']}")
    print(f"  → Search query: {state['search_query']}")
    return state


def node_web_search(state: PostState) -> PostState:
    """
    Node 2 – Web Search
    Executes mock_searxng_search with the query from Node 1.
    """
    print(f"\n[Node 2] Running mock web search: '{state['search_query']}' …")
    result = mock_searxng_search.invoke({"query": state["search_query"]})
    state["search_result"] = result
    print(f"  → Result: {result}")
    return state


def node_draft_post(state: PostState) -> PostState:
    """
    Node 3 – Draft Post
    The LLM uses Persona + Search Context to write a max-280-char post,
    returned as structured JSON.
    """
    print(f"\n[Node 3] Drafting post for {state['bot_id']} …")
    llm = get_llm(temperature=0.9)

    system_prompt = f"""You are {state['bot_id']}, a social media bot.
Your persona: "{state['persona']}"
You MUST always stay in character. Be opinionated, direct, and authentic to your persona.
Your post MUST be under 280 characters."""

    user_prompt = f"""Breaking news context: "{state['search_result']}"
Topic you chose: "{state['topic']}"

Write a single opinionated social media post about this topic using the news as context.
Reply ONLY with a JSON object (no markdown):
{{"bot_id": "{state['bot_id']}", "topic": "{state['topic']}", "post_content": "your post here"}}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    raw = response.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        parsed = json.loads(raw)
        # Enforce 280-char limit
        if len(parsed.get("post_content", "")) > 280:
            parsed["post_content"] = parsed["post_content"][:277] + "…"
        state["post_content"] = parsed["post_content"]
        state["final_json"]   = parsed
    except Exception as e:
        # Graceful fallback
        fallback = {
            "bot_id":       state["bot_id"],
            "topic":        state["topic"],
            "post_content": raw[:280],
        }
        state["post_content"] = raw[:280]
        state["final_json"]   = fallback

    print(f"  → Post ({len(state['post_content'])} chars): {state['post_content']}")
    return state


# ─────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────

def build_content_graph() -> StateGraph:
    """Compile and return the LangGraph state machine."""
    graph = StateGraph(PostState)

    graph.add_node("decide_search", node_decide_search)
    graph.add_node("web_search",    node_web_search)
    graph.add_node("draft_post",    node_draft_post)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search",    "draft_post")
    graph.add_edge("draft_post",    END)

    return graph.compile()


# ─────────────────────────────────────────────
# Demo / Entry Point
# ─────────────────────────────────────────────

def run_phase2_demo():
    from persona_router import BOT_PERSONAS, BOT_NAMES

    app = build_content_graph()

    print("=" * 65)
    print("PHASE 2 — Autonomous Content Engine Demo")
    print("=" * 65)

    results = []
    for bot_id, persona in BOT_PERSONAS.items():
        print(f"\n{'─'*55}")
        print(f"Running graph for [{bot_id.upper()}] {BOT_NAMES[bot_id]}")

        initial_state: PostState = {
            "bot_id":        bot_id,
            "persona":       persona,
            "search_query":  "",
            "search_result": "",
            "topic":         "",
            "post_content":  "",
            "final_json":    {},
        }

        final_state = app.invoke(initial_state)
        results.append(final_state["final_json"])

        print(f"\n  📦 Final JSON output:")
        print(f"  {json.dumps(final_state['final_json'], indent=2)}")

    print("\n" + "=" * 65)
    return results


if __name__ == "__main__":
    run_phase2_demo()
