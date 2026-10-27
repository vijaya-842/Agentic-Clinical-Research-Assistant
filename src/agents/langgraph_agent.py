"""Agentic RAG, step 2: an LLM-driven agent built with LangGraph.

Instead of fixed keyword rules, the LLM (qwen) chooses the tool. Then the agent
checks its own answer and, if it has no valid citations, tries another tool once.

    plan  ->  act  ->  check  --(ok)-->  END
                ^         |
                +-(retry)-+
"""
import json
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.router_agent import QUESTIONS_FILE, coverage, load_questions, route, trial_details

OUTPUT_FILE = Path("eval/langgraph_agent_results.json")
TOOL_NAMES = ["trials_then_papers", "count_trials", "list_trials", "search_trials", "search_papers"]
MAX_ATTEMPTS = 2

# If a tool's answer has no valid citations, which tool to try next
FALLBACK = {
    "trials_then_papers": "list_trials",
    "search_papers": "search_trials",
    "search_trials": "search_papers",
    "list_trials": "search_trials",
}

PLANNER_SYSTEM_PROMPT = """You choose ONE tool to answer a question about cancer clinical trials.
Tools:
- count_trials: the question asks HOW MANY trials (a number).
- list_trials: the question asks to find or list SEVERAL trials.
- search_trials: the question is about ONE specific trial (ID, sponsor, outcome, eligibility), or anything else about trials.
- search_papers: the question asks how a drug works (mechanism), side effects, resistance, or scientific evidence.
- trials_then_papers: the question asks to find trials AND explain how their drugs work (needs both).
Reply with ONLY the tool name, nothing else."""

PLANNER_EXAMPLES = """Examples:
Question: How many completed breast cancer trials are there? -> count_trials
Question: Find recruiting trials testing nivolumab in lung cancer. -> list_trials
Question: Who sponsors trial NCT01234567 and what is its primary outcome? -> search_trials
Question: How does trastuzumab work? -> search_papers
Question: Find recruiting breast cancer trials and explain how their drugs work. -> trials_then_papers
"""


class AgentState(TypedDict, total=False):
    question: str
    tool: str
    planner_output: str
    planner_fallback: bool
    attempts: int
    tried_tools: list
    result: Optional[dict]
    trace: list


def parse_tool_choice(text):
    """Find the first known tool name in the LLM's reply, or None."""
    text = text.lower()
    found = [(text.find(name), name) for name in TOOL_NAMES if name in text]
    return min(found)[1] if found else None


def answer_is_ok(result):
    """Good enough = cites at least one valid source, or is a proper refusal."""
    answer = result["answer"].lower()
    refused = "do not contain enough information" in answer or "consult" in answer
    return bool(result["valid_citations"]) or refused


RULE_TO_TOOL = {
    "safety_refusal": "safety_refusal",
    "trial_lookup": "trial_lookup",
    "count_trials": "count_trials",
    "list_rag": "list_trials",
    "advanced_rag": "search_trials",
    "literature_rag": "search_papers",
    "small_talk": "small_talk",
    "safety_refusal": "safety_refusal",
    "literature_rag": "search_papers",
    "multi_hop": "trials_then_papers",
}



def build_agent(tools, planner_llm, verifier=None):
    """tools: {tool name: function(question) -> result dict}. planner_llm: function(prompt, system) -> text.
    verifier: optional function(result, tool) -> result, run once at the end (critic / verifier step)."""

    def plan(state):
        # Safety guardrail and trial-ID lookup are decided by code BEFORE the LLM planner
        rule = route(state["question"])
        if rule in ("small_talk", "safety_refusal", "trial_lookup", "multi_hop"):
            tool = RULE_TO_TOOL[rule]
            return {"tool": tool, "planner_output": "", "planner_fallback": False, "attempts": 0,
                    "tried_tools": [], "trace": [f"plan: rule -> {tool} (LLM planner skipped)"]}
        prompt = f"{PLANNER_EXAMPLES}\nQuestion: {state['question']} ->"
        output = planner_llm(prompt, PLANNER_SYSTEM_PROMPT)
        tool = parse_tool_choice(output)
        fallback = tool is None
        if fallback:  # the LLM gave an unusable reply -> use the keyword rules instead
            tool = RULE_TO_TOOL[route(state["question"])]
        return {
            "tool": tool,
            "planner_output": output.strip(),
            "planner_fallback": fallback,
            "attempts": 0,
            "tried_tools": [],
            "trace": [f"plan: LLM said '{output.strip()[:40]}' -> {tool}" + (" (rule fallback)" if fallback else "")],
        }

    def act(state):
        tool = state["tool"]
        result = tools[tool](state["question"])
        best = state.get("result")
        # keep whichever answer has more valid citations
        if best is not None and len(best["valid_citations"]) > len(result["valid_citations"]):
            result = best
        return {
            "result": result,
            "attempts": state["attempts"] + 1,
            "tried_tools": state["tried_tools"] + [tool],
            "trace": state["trace"] + [f"act: {tool} -> {len(result['valid_citations'])} valid citations"],
        }

    def check(state):
        result = state["result"]
        if state["tool"] in ("count_trials", "small_talk", "safety_refusal", "trial_lookup") or answer_is_ok(result):
            return {"trace": state["trace"] + ["check: ok"]}
        next_tool = FALLBACK.get(state["tool"])
        if state["attempts"] < MAX_ATTEMPTS and next_tool and next_tool not in state["tried_tools"]:
            return {"tool": next_tool, "trace": state["trace"] + [f"check: no valid citations -> retry with {next_tool}"]}
        return {"trace": state["trace"] + ["check: no valid citations, no more retries"]}

    def verify(state):
        tool = state["tried_tools"][-1]
        if verifier is None:
            return {"trace": state["trace"] + ["verify: skipped (no verifier)"]}
        result = verifier(state["result"], tool)
        v = result.get("verification", {})
        line = f"verify: {v.get('status')}"
        if v.get("issues"):
            line += f" ({len(v['issues'])} issue(s))"
        return {"result": result, "trace": state["trace"] + [line]}

    def after_check(state):
        return "retry" if state["tool"] not in state["tried_tools"] else "done"

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("act", act)
    graph.add_node("check", check)
    graph.add_node("verify", verify)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "check")
    graph.add_conditional_edges("check", after_check, {"retry": "act", "done": "verify"})
    graph.add_edge("verify", END)
    return graph.compile()



def make_tools(retriever, trial_facts, all_trials, pubmed_retriever):
    """Wrap our existing tools so each one takes just the question."""
    from src.agents.multi_hop import answer_multi_hop
    from src.agents.router_agent import LIST_TOP_K, answer_with_count_tool, list_with_details, safety_refusal_result
    from src.tools.small_talk import small_talk_result
    from src.tools.trial_lookup import answer_trial_lookup
    from src.rag.advanced_rag import answer_question as advanced_answer
    from src.rag.literature_rag import answer_literature_question

    def list_trials(question):
        return list_with_details(advanced_answer(question, retriever, trial_facts, top_k=LIST_TOP_K), all_trials)

    def search_papers(question):
        return answer_literature_question(question, pubmed_retriever, top_k=5)

    return {
        "small_talk": small_talk_result,
        "safety_refusal": safety_refusal_result,
        "trial_lookup": lambda q: answer_trial_lookup(q, all_trials),
        "count_trials": lambda q: answer_with_count_tool(q, all_trials),
        "list_trials": list_trials,
        "search_trials": lambda q: advanced_answer(q, retriever, trial_facts, top_k=5),
        "search_papers": search_papers,
        "trials_then_papers": lambda q: answer_multi_hop(q, list_trials, search_papers, all_trials),
    }

def run_evaluation(output_path=OUTPUT_FILE, questions_path=QUESTIONS_FILE, rerank=True, hybrid=True):
    from src.llm.ollama_client import generate_response
    from src.rag.trial_facts import load_trial_facts
    from src.retrieval.pipeline import load_retrievers
    from src.tools.count_trials import load_trials

    print("Loading indexes, models and trial data...")
    all_trials = load_trials()
    retriever, pubmed_retriever = load_retrievers(hybrid=hybrid, rerank=rerank)
    tools = make_tools(retriever, load_trial_facts(), all_trials, pubmed_retriever)
    from src.agents.verifier import verify_answer
    agent = build_agent(tools, generate_response,
                        verifier=lambda result, tool: verify_answer(result, all_trials, tool))

    results = []
    for q in load_questions(questions_path):
        print(f"\n{q['id']}: {q['question']}")
        state = agent.invoke({"question": q["question"]})
        result = state["result"]
        gold = q.get("gold_nct_ids", [])
        result.update({
            "id": q["id"],
            "category": q.get("category"),
            "tool": state["tried_tools"][-1],
            "tools_tried": state["tried_tools"],
            "planner_output": state["planner_output"],
            "planner_fallback": state["planner_fallback"],
            "trace": state["trace"],
            "gold_in_retrieved": coverage(gold, result["retrieved_nct_ids"]),
            "gold_in_answer": coverage(gold, result["cited_nct_ids"]),
            "verification": result.get("verification"),
        })
        for step in state["trace"]:
            print("  " + step)
        print(f"Answer: {result['answer']}")
        results.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(QUESTIONS_FILE))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--no-hybrid", action="store_true", help="dense search only (no BM25)")
    args = parser.parse_args()
    run_evaluation(output_path=Path(args.output), questions_path=Path(args.questions), hybrid=not args.no_hybrid)