"""Streamlit UI for the Agentic Clinical Research Assistant.

Run from the project root (with Ollama running):
    streamlit run src/ui/app.py
"""
import sys
import time
from pathlib import Path

import streamlit as st

# Streamlit runs this file directly, so make "from src..." imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TOOL_LABELS = {
    "safety_refusal": "Safety guardrail (no search, no LLM)",
    "trial_lookup": "Trial lookup by ID (code)",
    "count_trials": "Count tool (code counts all trials)",
    "list_rag": "Trial search: list mode (10 trials)",
    "list_trials": "Trial search: list mode (10 trials)",
    "advanced_rag": "Trial search (5 trials + fact sheets)",
    "search_trials": "Trial search (5 trials + fact sheets)",
    "literature_rag": "PubMed paper search",
    "search_papers": "PubMed paper search",
}

EXAMPLES = [
    "Type your own question...",
    "How many recruiting pancreatic cancer trials are in the dataset?",
    "Find recruiting clinical trials testing pembrolizumab in melanoma.",
    "Tell me about NCT05086614.",
    "What is the mechanism of action of osimertinib?",
    "Why do EGFR inhibitors like osimertinib eventually stop working?",
    "What's the total of phase 3 lung cancer trials in our dataset?",
    "My father has stage IV lung cancer. Which treatment should he take?",
]


@st.cache_resource(show_spinner="Loading indexes and models (first time only, about 1 minute)...")
def load_resources():
    from src.rag.trial_facts import load_trial_facts
    from src.reranking.cross_encoder import RerankingRetriever
    from src.retrieval.advanced import AdvancedRetriever
    from src.retrieval.pubmed_search import PubMedRetriever
    from src.tools.count_trials import load_trials

    reranker = RerankingRetriever.load_model()
    return {
        "retriever": RerankingRetriever(AdvancedRetriever.from_disk(), reranker),
        "pubmed_retriever": RerankingRetriever(PubMedRetriever.from_disk(), reranker),
        "trial_facts": load_trial_facts(),
        "all_trials": load_trials(),
    }


@st.cache_resource(show_spinner=False)
def load_langgraph_agent():
    from src.agents.langgraph_agent import build_agent, make_tools
    from src.llm.ollama_client import generate_response

    r = load_resources()
    tools = make_tools(r["retriever"], r["trial_facts"], r["all_trials"], r["pubmed_retriever"])
    return build_agent(tools, generate_response)


def ask(question, agent_name):
    """Run the chosen agent. Returns (result dict, trace lines)."""
    r = load_resources()
    if agent_name == "Keyword router":
        from src.agents.router_agent import answer_question
        result = answer_question(question, r["retriever"], r["trial_facts"], r["all_trials"],
                                 pubmed_retriever=r["pubmed_retriever"])
        return result, [f"rule-based route -> {result['tool']}"]
    state = load_langgraph_agent().invoke({"question": question})
    result = state["result"]
    result["tool"] = state["tried_tools"][-1]
    return result, state["trace"]


def show_sources(result):
    trials = [c for c in result.get("valid_citations", []) if c.startswith("NCT")]
    papers = result.get("cited_pmids", [])
    if not trials and not papers:
        return
    st.subheader("Sources")
    for nct in trials:
        st.markdown(f"- Trial [{nct}](https://clinicaltrials.gov/study/{nct})")
    for pmid in papers:
        st.markdown(f"- Paper [PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
    if result.get("invented_citations"):
        st.warning("Removed / unsupported citations: " + ", ".join(result["invented_citations"]))


# ---------------- page ----------------
st.set_page_config(page_title="Clinical Research Assistant", page_icon="🔬", layout="wide")
st.title("🔬 Agentic Clinical Research Assistant")
st.caption("Oncology trials (ClinicalTrials.gov, 4,407 trials) and PubMed papers (1,592 abstracts). "
           "Local models only: qwen2.5:3b, bge-base, MedCPT.")

with st.sidebar:
    agent_name = st.radio("Agent", ["LangGraph agent (LLM picks tool)", "Keyword router"])
    st.markdown("**Tools**\n- Safety guardrail\n- Trial lookup by ID\n- Count tool\n"
                "- Trial search (filters + reranker)\n- PubMed paper search")
    st.info("Research tool only. Not medical advice.")

example = st.selectbox("Example questions", EXAMPLES)
default = "" if example == EXAMPLES[0] else example
question = st.text_area("Your question", value=default, height=80)

if st.button("Ask", type="primary") and question.strip():
    start = time.time()
    with st.spinner("Thinking... (answers can take 30-90 seconds on CPU)"):
        result, trace = ask(question.strip(), agent_name.split(" (")[0])
    seconds = time.time() - start

    tool = result.get("tool", "")
    col1, col2 = st.columns([3, 1])
    col1.markdown(f"**Tool used:** {TOOL_LABELS.get(tool, tool)}")
    col2.markdown(f"**Time:** {seconds:.0f} s")

    st.subheader("Answer")
    st.markdown(result["answer"])
    show_sources(result)

    with st.expander("How the agent decided (trace)"):
        for line in trace:
            st.text(line)