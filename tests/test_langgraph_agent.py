from src.agents.langgraph_agent import answer_is_ok, build_agent, parse_tool_choice


def fake_result(valid=(), answer="Some answer"):
    return {"answer": answer, "valid_citations": list(valid), "retrieved_nct_ids": [], "cited_nct_ids": []}


def make_tools(papers_ok=True, trials_ok=True):
    calls = []

    def tool(name, ok):
        def run(question):
            calls.append(name)
            return fake_result(valid=["X1"] if ok else [])
        return run

    tools = {
        "count_trials": tool("count_trials", True),
        "list_trials": tool("list_trials", trials_ok),
        "search_trials": tool("search_trials", trials_ok),
        "search_papers": tool("search_papers", papers_ok),
    }
    return tools, calls


def planner_says(text):
    return lambda prompt, system_prompt: text


def test_parse_tool_choice():
    assert parse_tool_choice("search_papers") == "search_papers"
    assert parse_tool_choice("I would use count_trials here.") == "count_trials"
    assert parse_tool_choice("no idea") is None


def test_answer_is_ok():
    assert answer_is_ok(fake_result(valid=["NCT1"]))
    assert answer_is_ok(fake_result(answer="Please consult his oncologist."))
    assert not answer_is_ok(fake_result())


def test_llm_choice_is_used():
    tools, calls = make_tools()
    agent = build_agent(tools, planner_says("search_papers"))
    state = agent.invoke({"question": "How does osimertinib work?"})
    assert calls == ["search_papers"]
    assert state["planner_fallback"] is False


def test_unusable_llm_reply_falls_back_to_rules():
    tools, calls = make_tools()
    agent = build_agent(tools, planner_says("hmm, not sure"))
    state = agent.invoke({"question": "How many recruiting pancreatic cancer trials are there?"})
    assert calls == ["count_trials"]
    assert state["planner_fallback"] is True


def test_retries_with_other_tool_when_no_citations():
    tools, calls = make_tools(papers_ok=False, trials_ok=True)
    agent = build_agent(tools, planner_says("search_papers"))
    state = agent.invoke({"question": "Some question"})
    assert calls == ["search_papers", "search_trials"]
    assert state["result"]["valid_citations"] == ["X1"]


def test_stops_after_max_attempts():
    tools, calls = make_tools(papers_ok=False, trials_ok=False)
    agent = build_agent(tools, planner_says("search_papers"))
    agent.invoke({"question": "Some question"})
    assert calls == ["search_papers", "search_trials"]

def test_guardrail_runs_before_llm_planner():
    tools, calls = make_tools()
    tools["safety_refusal"] = lambda q: fake_result(answer="Please consult your oncologist.")
    agent = build_agent(tools, planner_says("search_papers"))
    state = agent.invoke({"question": "I have breast cancer. Should I stop my chemotherapy?"})
    assert state["tried_tools"] == ["safety_refusal"]
    assert calls == []

def test_verifier_runs_at_the_end():
    tools, calls = make_tools()

    def fake_verifier(result, tool):
        result["verification"] = {"status": "pass", "issues": []}
        return result

    agent = build_agent(tools, planner_says("search_papers"), verifier=fake_verifier)
    state = agent.invoke({"question": "How does osimertinib work?"})
    assert state["result"]["verification"]["status"] == "pass"
    assert state["trace"][-1] == "verify: pass"


def test_multi_hop_is_decided_by_rule():
    tools, calls = make_tools()
    tools["trials_then_papers"] = lambda q: fake_result(valid=["NCT1", "PMID 1"])
    agent = build_agent(tools, planner_says("search_trials"))
    state = agent.invoke({"question": "Which recruiting melanoma trials test pembrolizumab, and how does pembrolizumab work?"})
    assert state["tried_tools"] == ["trials_then_papers"]