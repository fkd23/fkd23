"""Ask a question and get a web-grounded answer with its sources.

Uses the Perplexity Agent API (``POST https://api.perplexity.ai/v1/agent``) through the
official SDK. The API key is read only from the ``PERPLEXITY_API_KEY`` environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import perplexity
from perplexity import Perplexity

API_KEY_ENV = "PERPLEXITY_API_KEY"
DEFAULT_EFFORT = "low"


class AnswerError(Exception):
    """Raised when an answer could not be produced. The message is safe to show to users."""


class MissingApiKeyError(AnswerError):
    """Raised when PERPLEXITY_API_KEY is not set."""


@dataclass(frozen=True)
class Source:
    """One web result the answer was grounded on (``search_results`` item)."""

    id: int | None
    url: str
    title: str | None = None


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    response_id: str | None = None
    effort: str | None = None
    model: str | None = None


def _extract_sources(response: Any) -> list[Source]:
    """Collect sources from ``search_results`` output items, then URL citation annotations."""
    sources: list[Source] = []
    seen: set[str] = set()

    def add(source_id: int | None, url: str | None, title: str | None) -> None:
        if url and url not in seen:
            seen.add(url)
            sources.append(Source(id=source_id, url=url, title=title))

    for item in getattr(response, "output", None) or []:
        item_type = getattr(item, "type", None)
        if item_type == "search_results":
            for result in getattr(item, "results", None) or []:
                add(
                    getattr(result, "id", None),
                    getattr(result, "url", None),
                    getattr(result, "title", None),
                )
        elif item_type == "message":
            for part in getattr(item, "content", None) or []:
                for annotation in getattr(part, "annotations", None) or []:
                    if getattr(annotation, "type", None) == "url_citation":
                        add(
                            None,
                            getattr(annotation, "url", None),
                            getattr(annotation, "title", None),
                        )
    return sources


def _build_client(client: Perplexity | None) -> Perplexity:
    if client is not None:
        return client
    if not os.environ.get(API_KEY_ENV):
        raise MissingApiKeyError(
            f"{API_KEY_ENV} is not set. Create a key in the API Console "
            "(https://console.perplexity.ai) and export it in your own terminal: "
            f"export {API_KEY_ENV}=..."
        )
    return Perplexity()  # reads PERPLEXITY_API_KEY itself; retries 429/5xx by default


def ask(
    question: str,
    *,
    effort: str | None = None,
    model: str | None = None,
    previous_response_id: str | None = None,
    instructions: str | None = None,
    client: Perplexity | None = None,
) -> Answer:
    """Return a web-grounded answer.

    Pass ``effort`` (``fast|low|medium|high|xhigh|ultra``, default ``low``) to use a
    Perplexity-managed configuration, or ``model`` (e.g. ``"openai/gpt-5.6-sol"``) to choose
    the model yourself; with ``model`` the ``web_search`` tool is added explicitly.
    Pass ``previous_response_id`` (an ``Answer.response_id``) to continue a conversation.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if effort and model:
        raise ValueError("pass either effort or model, not both")

    params: dict[str, Any] = {"input": question}
    if model:
        params["model"] = model
        params["tools"] = [{"type": "web_search"}]
    else:
        params["effort"] = effort or DEFAULT_EFFORT
    if previous_response_id:
        params["previous_response_id"] = previous_response_id
    if instructions:
        params["instructions"] = instructions

    api = _build_client(client)
    try:
        response = api.responses.create(**params)
    except perplexity.AuthenticationError as exc:
        raise AnswerError(
            f"Perplexity rejected the API key (401). Check {API_KEY_ENV} in the API Console "
            "(https://console.perplexity.ai)."
        ) from exc
    except perplexity.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after")
        wait = f" Retry after {retry_after}s." if retry_after else ""
        raise AnswerError(f"Perplexity rate limit or overload (429).{wait}") from exc
    except perplexity.APIConnectionError as exc:
        raise AnswerError(f"Could not reach the Perplexity API: {exc}") from exc
    except perplexity.APIStatusError as exc:
        raise AnswerError(f"Perplexity API error {exc.status_code}: {exc.message}") from exc

    return Answer(
        text=response.output_text or "",
        sources=_extract_sources(response),
        response_id=getattr(response, "id", None),
        effort=getattr(response, "effort", None),
        model=getattr(response, "model", None),
    )
