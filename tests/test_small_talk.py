from src.tools.small_talk import small_talk_reply, small_talk_result


def test_greetings_get_a_reply():
    for text in ["hi", "Hi!", "hello", "Hey 👋", "good morning", "HIII"]:
        assert small_talk_reply(text) is not None, text


def test_thanks_bye_and_help():
    assert small_talk_reply("thanks!").startswith("You're welcome")
    assert small_talk_reply("Bye").startswith("Goodbye")
    assert "Tell me about NCT05086614" in small_talk_reply("what can you do?")


def test_real_questions_are_not_small_talk():
    assert small_talk_reply("hi, how many melanoma trials are there?") is None
    assert small_talk_reply("Tell me about NCT05086614.") is None
    assert small_talk_reply("") is None


def test_result_has_no_citations():
    result = small_talk_result("hello")
    assert result["valid_citations"] == []
    assert result["answer"].startswith("Hello!")