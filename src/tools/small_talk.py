"""Small talk: greetings, thanks and "what can you do" get a fixed reply.

No search, no LLM -> instant answer and no wasted compute.
Only SHORT messages that are nothing but small talk match;
"hi, how many melanoma trials are there?" still goes to the real tools.
"""
import re

GREETING = re.compile(r"^(hi+|hello+|hey+|hiya|good (morning|afternoon|evening)|greetings)$")
THANKS = re.compile(r"^(thanks?( you)?( so much| a lot)?|thx|ty|great,? thanks?|ok,? thanks?)$")
BYE = re.compile(r"^(bye|goodbye|see you|see ya|good night)$")
HELP = re.compile(r"^(help|what can you do|who are you|what are you|how do i use this)$")

CAPABILITIES = (
    "I can help with oncology clinical-trial research, for example:\n"
    "- Find trials: \"Find recruiting Phase 3 lung cancer trials\"\n"
    "- Look up a trial: \"Tell me about NCT05086614\"\n"
    "- Count trials: \"How many recruiting pancreatic cancer trials are there?\"\n"
    "- Explain a drug from research papers: \"What is the mechanism of action of osimertinib?\"\n"
    "I can't give personal medical advice."
)

REPLIES = [
    (GREETING, "Hello! " + CAPABILITIES),
    (THANKS, "You're welcome! Ask me another question about cancer trials or research papers any time."),
    (BYE, "Goodbye! Come back any time you have a question about cancer trials."),
    (HELP, CAPABILITIES),
]


def clean(text):
    """Lowercase, drop punctuation and emojis, squeeze spaces: 'Hi!! ' -> 'hi'."""
    text = re.sub(r"[^a-z\s,]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip(" ,")


def small_talk_reply(question):
    """Return a fixed reply if the whole message is small talk, else None."""
    text = clean(question)
    if not text or len(text.split()) > 6:
        return None
    for pattern, reply in REPLIES:
        if pattern.match(text):
            return reply
    return None


def small_talk_result(question):
    """Same result shape as the other tools."""
    return {
        "question": question,
        "answer": small_talk_reply(question) or CAPABILITIES,
        "retrieved_chunk_ids": [],
        "retrieved_nct_ids": [],
        "cited_nct_ids": [],
        "valid_citations": [],
        "invented_citations": [],
        "retrieval_seconds": 0.0,
        "generation_seconds": 0.0,
        "filters": {},
    }