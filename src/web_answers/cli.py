"""Command line entry point: ``web-answers "question"``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .answers import DEFAULT_EFFORT, AnswerError, ask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="web-answers", description=__doc__)
    parser.add_argument("question")
    parser.add_argument(
        "--effort",
        default=None,
        help=f"fast|low|medium|high|xhigh|ultra (default {DEFAULT_EFFORT})",
    )
    parser.add_argument("--model", default=None, help="e.g. openai/gpt-5.6-sol (uses web_search)")
    parser.add_argument("--previous-response-id", default=None, help="continue a conversation")
    parser.add_argument("--json", action="store_true", help="print the full answer as JSON")
    args = parser.parse_args(argv)

    try:
        answer = ask(
            args.question,
            effort=args.effort,
            model=args.model,
            previous_response_id=args.previous_response_id,
        )
    except (AnswerError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(answer), ensure_ascii=False, indent=2))
    else:
        print(answer.text)
        if answer.sources:
            print("\nSources:")
            for s in answer.sources:
                print(f"  [{s.id if s.id is not None else '-'}] {s.title or s.url} - {s.url}")
        if answer.response_id:
            print(f"\nresponse id: {answer.response_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
