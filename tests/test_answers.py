from types import SimpleNamespace as NS

import httpx
import perplexity
import pytest

from web_answers import AnswerError, MissingApiKeyError, ask


def _response():
    return NS(
        id="resp_1",
        model="m",
        effort="low@2026-01-01.0",
        output_text="Paris [1].",
        output=[
            NS(
                type="search_results",
                results=[
                    NS(id=1, url="https://a.example", title="A"),
                    NS(id=2, url="https://b.example", title="B"),
                ],
            ),
            NS(
                type="message",
                content=[
                    NS(
                        annotations=[
                            NS(type="url_citation", url="https://a.example", title="A"),
                            NS(type="url_citation", url="https://c.example", title="C"),
                        ]
                    )
                ],
            ),
        ],
    )


class FakeClient:
    def __init__(self, response=None, error=None):
        self.calls = []
        self._response, self._error = response, error
        self.responses = NS(create=self._create)

    def _create(self, **params):
        self.calls.append(params)
        if self._error:
            raise self._error
        return self._response


def _status_error(cls, status, headers=None):
    request = httpx.Request("POST", "https://api.perplexity.ai/v1/agent")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("boom", response=response, body=None)


def test_effort_default_and_sources_deduped():
    client = FakeClient(_response())
    answer = ask("capital of France?", client=client)
    assert client.calls == [{"input": "capital of France?", "effort": "low"}]
    assert answer.text == "Paris [1]."
    assert [s.url for s in answer.sources] == [
        "https://a.example",
        "https://b.example",
        "https://c.example",
    ]
    assert answer.response_id == "resp_1"


def test_model_adds_web_search_tool_and_conversation_id():
    client = FakeClient(_response())
    ask("q", model="openai/gpt-5.6-sol", previous_response_id="resp_0", client=client)
    assert client.calls[0] == {
        "input": "q",
        "model": "openai/gpt-5.6-sol",
        "tools": [{"type": "web_search"}],
        "previous_response_id": "resp_0",
    }


def test_effort_and_model_are_exclusive():
    with pytest.raises(ValueError):
        ask("q", effort="low", model="x", client=FakeClient())


def test_empty_question():
    with pytest.raises(ValueError):
        ask("  ", client=FakeClient())


def test_missing_key(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError, match="console.perplexity.ai"):
        ask("q")


def test_auth_error_message_hides_nothing_sensitive():
    err = _status_error(perplexity.AuthenticationError, 401)
    with pytest.raises(AnswerError, match="401"):
        ask("q", client=FakeClient(error=err))


def test_rate_limit_reports_retry_after():
    err = _status_error(perplexity.RateLimitError, 429, {"retry-after": "7"})
    with pytest.raises(AnswerError, match="Retry after 7s"):
        ask("q", client=FakeClient(error=err))
