"""Prompt rendering; expected strings are shared with rust/deem-runtime/src/format.rs."""

import pytest

from deem.format import build_prompt, render_inline, render_text
from deem.primitives import (
    ChoiceQuestion,
    DeemError,
    NoulQuestion,
    QuestionSet,
    ScoreQuestion,
)


def test_prompt_layout():
    qs = QuestionSet({"q": ChoiceQuestion("What to do?", ["deploy", "hold"])})
    assert build_prompt("deploy box", qs) == (
        '<state>\n"deploy box"\n</state>\n\n'
        "Question 1: What to do?\nOptions:\n(A) deploy\n(B) hold\nAnswer 1: ("
    )


def test_prompt_layout_with_criteria():
    qs = QuestionSet(
        {
            "team": ChoiceQuestion(
                "Which team?",
                ["billing", "sales"],
                descriptions=["Payments, refunds", None],
            ),
            "urgent": NoulQuestion(
                "Is it urgent?", if_true="Time-sensitive", if_false="No urgency"
            ),
        }
    )
    prompt = build_prompt("payouts failing", qs, {"team": [1, 0]})
    assert prompt == (
        '<state>\n"payouts failing"\n</state>\n\n'
        "Question 1: Which team?\nOptions:\n(A) sales\n(B) billing: Payments, refunds\n"
        "Answer 1: (\n\n"
        "Question 2: Is it urgent?\nTrue if: Time-sensitive\nFalse if: No urgency\n"
        "Answer 2: ("
    )


def test_noul_single_criterion_and_empty_instructions():
    qs = QuestionSet({"q": NoulQuestion("", if_false="never")})
    assert build_prompt("s", qs).endswith("Question 1: \nFalse if: never\nAnswer 1: (")


def test_all_none_descriptions_render_bare():
    plain = QuestionSet({"q": ChoiceQuestion("pick", ["a", "b"])})
    described = QuestionSet(
        {"q": ChoiceQuestion("pick", ["a", "b"], descriptions=[None, None])}
    )
    assert build_prompt("s", plain) == build_prompt("s", described)


def test_non_string_values_render_as_canonical_json():
    assert render_text({"b": 1, "a": "<x>"}) == '{"a":"\\u003cx\\u003e","b":1}'
    assert render_text("line\nbreak <ok>") == "line\nbreak <ok>"
    assert render_inline("a\r\nb\nc") == "a  b c"
    assert render_inline(["é", None]) == '["é",null]'
    qs = QuestionSet(
        {
            "q": ChoiceQuestion(
                "pick", ["x", "y"], descriptions=[{"z": 1, "a": [2]}, "multi\nline"]
            )
        }
    )
    assert '(A) x: {"a":[2],"z":1}\n(B) y: multi line\n' in build_prompt("s", qs)


def test_canonical_json_sorts_nested_keys():
    value = {"b": {"z": 1, "a": [{"y": 2, "x": "<"}]}, "a": 0}
    assert render_text(value) == '{"a":0,"b":{"a":[{"x":"\\u003c","y":2}],"z":1}}'


def test_descriptions_must_match_options():
    with pytest.raises(DeemError):
        ChoiceQuestion("pick", ["a", "b"], descriptions=["only one"])
    with pytest.raises(TypeError):
        ChoiceQuestion("pick", ["a", "b"], descriptions="ab")


def test_score_prompt_unchanged():
    qs = QuestionSet({"q": ScoreQuestion("rate", ["low", "high"])})
    assert build_prompt(1, qs) == (
        "<state>\n1\n</state>\n\nQuestion 1: rate\nOptions:\n(A) low\n(B) high\n"
        "Answer 1: ("
    )
