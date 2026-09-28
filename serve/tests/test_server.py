"""HTTP contract of the /v1/systemone server (stub backends only)."""

import json
import math

import pytest
from serve_helpers import FixedBackend, get, live_server, post


def systemone_url():
    return "/v1/systemone"


def simple_payload(**overrides):
    payload = {
        "state": "The build is red on a Friday evening.",
        "questions": {
            "deploy": {
                "type": "choice",
                "instructions": "Deploy now or wait?",
                "options": ["deploy", "wait"],
            }
        },
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Request validation: 422 with FastAPI-style detail
# ---------------------------------------------------------------------------


def assert_422(status, body, loc=None, typ="value_error"):
    assert status == 422
    (error,) = body["detail"]
    assert error["type"] == typ
    assert isinstance(error["msg"], str)
    assert error["loc"][0] == "body"
    if loc is not None:
        assert error["loc"] == loc


def one_question(spec):
    return {"state": "x", "questions": {"q": spec}}


def test_missing_state_422(base_url):
    status, body = post(
        base_url, systemone_url(), {"questions": simple_payload()["questions"]}
    )
    assert_422(status, body, ["body", "state"])


def test_missing_questions_422(base_url):
    status, body = post(base_url, systemone_url(), {"state": "hello"})
    assert_422(status, body, ["body", "questions"])


def test_empty_questions_422(base_url):
    status, body = post(base_url, systemone_url(), {"state": "x", "questions": {}})
    assert_422(status, body, ["body", "questions"])


def test_malformed_json_422(base_url):
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        base_url + systemone_url(),
        data=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=30)
        raise AssertionError("expected 422")
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read())
        assert_422(exc.code, body, ["body"], typ="json_invalid")


def test_body_too_large_413(base_url):
    import http.client
    from urllib.parse import urlparse

    from deem_server import MAX_BODY_BYTES

    conn = http.client.HTTPConnection(urlparse(base_url).netloc, timeout=30)
    conn.putrequest("POST", systemone_url())
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
    conn.endheaders()
    response = conn.getresponse()
    assert response.status == 413
    body = json.loads(response.read())
    conn.close()
    assert body["detail"]["error_type"] == "request_too_large"
    assert body["detail"]["message"]


def test_one_option_choice_422(base_url):
    status, body = post(
        base_url,
        systemone_url(),
        one_question(
            {"type": "choice", "instructions": "pick", "criteria": {"only": None}}
        ),
    )
    assert_422(status, body, ["body", "questions", "q", "criteria"])


def test_one_option_legacy_options_422(base_url):
    status, body = post(
        base_url,
        systemone_url(),
        one_question({"type": "choice", "instructions": "pick", "options": ["one"]}),
    )
    assert_422(status, body, ["body", "questions", "q", "criteria"])


def test_256_options_choice_422(base_url):
    criteria = {f"option {i}": None for i in range(256)}
    status, body = post(
        base_url,
        systemone_url(),
        one_question({"type": "choice", "instructions": "pick", "criteria": criteria}),
    )
    assert_422(status, body, ["body", "questions", "q", "criteria"])


def test_255_options_ok(base_url):
    criteria = {f"option {i}": None for i in range(255)}
    status, body = post(
        base_url,
        systemone_url(),
        one_question({"type": "choice", "instructions": "pick", "criteria": criteria}),
    )
    assert status == 200
    assert len(body["answers"]["q"]["probabilities"]) == 255


def test_eleven_levels_422(base_url):
    status, body = post(
        base_url,
        systemone_url(),
        one_question(
            {
                "type": "score",
                "instructions": "rate",
                "criteria": [f"L{i}" for i in range(11)],
            }
        ),
    )
    assert_422(status, body, ["body", "questions", "q", "criteria"])


def test_single_level_422(base_url):
    status, body = post(
        base_url,
        systemone_url(),
        one_question({"type": "score", "instructions": "rate", "levels": ["high"]}),
    )
    assert_422(status, body, ["body", "questions", "q", "criteria"])


def test_unknown_question_type_422(base_url):
    status, body = post(
        base_url, systemone_url(), one_question({"type": "vibe", "instructions": "?"})
    )
    assert_422(status, body, ["body", "questions", "q", "type"])
    assert "vibe" in body["detail"][0]["msg"]


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]},
        {"type": "choice", "instructions": "pick"},
        {"type": "choice", "instructions": "pick", "criteria": {"a": None, " ": None}},
        {
            "type": "choice",
            "instructions": "pick",
            "criteria": {"a": None, "b\nc": None},
        },
        {"type": "score", "instructions": "rate", "criteria": {"a": 1, "b": 2}},
        {"type": "score", "instructions": "rate", "criteria": ["same", "same"]},
        {"type": "score", "instructions": "rate"},
        {"type": "noul", "instructions": "prop", "criteria": ["yes", "no"]},
    ],
)
def test_bad_criteria_422(base_url, spec):
    status, body = post(base_url, systemone_url(), one_question(spec))
    assert_422(status, body, ["body", "questions", "q", "criteria"])


def test_legacy_non_string_options_422(base_url):
    status, body = post(
        base_url,
        systemone_url(),
        one_question({"type": "choice", "instructions": "pick", "options": ["a", 2]}),
    )
    assert_422(status, body, ["body", "questions", "q", "options"])


def test_bad_instructions_type_422(base_url):
    status, body = post(
        base_url, systemone_url(), one_question({"type": "noul", "instructions": 3})
    )
    assert_422(status, body, ["body", "questions", "q", "instructions"])


def test_duplicate_options_422(base_url):
    status, body = post(
        base_url,
        systemone_url(),
        one_question(
            {"type": "choice", "instructions": "pick", "options": ["same", "same"]}
        ),
    )
    assert_422(status, body, ["body", "questions", "q", "criteria"])


def test_missing_instructions_ok(base_url):
    backend = FixedBackend()
    with live_server(backend=backend) as url:
        status, body = post(
            url,
            systemone_url(),
            one_question({"type": "noul", "criteria": {"true": "yes", "false": "no"}}),
        )
    assert status == 200
    assert body["answers"]["q"]["type"] == "noul"
    assert backend.calls[0][0].endswith(
        "Question 1: \nTrue if: yes\nFalse if: no\nAnswer 1: ("
    )


def test_question_count_cap():
    questions = {
        f"q{i}": {"type": "noul", "instructions": f"prop {i}"} for i in range(65)
    }
    with live_server() as url:
        status, body = post(
            url, systemone_url(), {"state": "x", "questions": questions}
        )
        assert_422(status, body, ["body", "questions"])
        assert "64" in body["detail"][0]["msg"]


def test_backend_letter_cap_maps_to_422():
    class CappedBackend(FixedBackend):
        max_letters = 26

    backend = CappedBackend()
    with live_server(backend=backend) as url:
        status, body = post(
            url,
            systemone_url(),
            one_question(
                {
                    "type": "choice",
                    "instructions": "pick",
                    "criteria": {f"o{i}": None for i in range(27)},
                }
            ),
        )
        assert_422(status, body, ["body", "questions", "q", "criteria"])
        assert "26" in body["detail"][0]["msg"]
        assert backend.calls == []


# ---------------------------------------------------------------------------
# TypeSafe criteria
# ---------------------------------------------------------------------------


def test_typesafe_criteria_request():
    backend = FixedBackend()
    payload = {
        "model": "jev-latest",
        "state": "Help! My payouts have been failing for 3 days.",
        "questions": {
            "urgent": {
                "type": "noul",
                "instructions": "Urgent?",
                "criteria": {"true": "Time-sensitive", "false": "No urgency"},
            },
            "team": {
                "type": "choice",
                "instructions": {"question": "Which team?"},
                "criteria": {"technical": "Bugs", "billing": None, "sales": {"x": 1}},
            },
            "mood": {
                "type": "score",
                "instructions": "Mood?",
                "criteria": ["Calm", {"level": "Frustrated"}, "Very\nangry"],
            },
        },
    }
    with live_server(backend=backend) as url:
        status, body = post(url, systemone_url(), payload)
    assert status == 200
    assert list(body["answers"]) == ["urgent", "team", "mood"]
    urgent, team, mood = (c[0] for c in backend.calls)
    assert urgent.endswith(
        "Question 1: Urgent?\nTrue if: Time-sensitive\nFalse if: No urgency\n"
        "Answer 1: ("
    )
    assert team.endswith(
        'Question 1: {"question":"Which team?"}\nOptions:\n'
        '(A) technical: Bugs\n(B) billing\n(C) sales: {"x":1}\nAnswer 1: ('
    )
    assert mood.endswith(
        'Options:\n(A) Calm\n(B) {"level":"Frustrated"}\n(C) Very angry\nAnswer 1: ('
    )
    assert list(body["answers"]["team"]["probabilities"]) == [
        "technical",
        "billing",
        "sales",
    ]
    assert body["answers"]["mood"]["legend"] == {
        "0": "Calm",
        "1": '{"level":"Frustrated"}',
        "2": "Very angry",
    }


def test_legacy_and_criteria_render_identically():
    backend = FixedBackend()
    legacy = one_question(
        {"type": "choice", "instructions": "pick", "options": ["a", "b"]}
    )
    typesafe = one_question(
        {"type": "choice", "instructions": "pick", "criteria": {"a": None, "b": None}}
    )
    with live_server(backend=backend) as url:
        assert post(url, systemone_url(), legacy)[0] == 200
        assert post(url, systemone_url(), typesafe)[0] == 200
    assert backend.calls[0] == backend.calls[1]


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


def test_choice_response_shape(base_url):
    status, body = post(base_url, systemone_url(), simple_payload())
    assert status == 200
    assert set(body) == {"model", "answers", "usage"}
    assert body["model"] == "deem-test"
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    assert body["usage"]["output_tokens"] == 0
    answer = body["answers"]["deploy"]
    assert set(answer) == {
        "type",
        "choice",
        "probabilities",
        "confidence",
        "x_temperature",
    }
    assert answer["type"] == "choice"
    assert answer["choice"] in ("deploy", "wait")
    assert set(answer["probabilities"]) == {"deploy", "wait"}
    assert answer["confidence"] == 0.0  # uniform stub
    assert answer["x_temperature"] == 1.0


def test_score_response_shape(base_url):
    status, body = post(
        base_url,
        systemone_url(),
        {
            "state": "The support ticket.",
            "questions": {
                "urgency": {
                    "type": "score",
                    "instructions": "Rate the urgency.",
                    "criteria": ["low", "medium", "high", "critical"],
                }
            },
        },
    )
    assert status == 200
    answer = body["answers"]["urgency"]
    assert set(answer) == {
        "type",
        "score",
        "legend",
        "probabilities",
        "confidence",
        "x_temperature",
    }
    assert answer["type"] == "score"
    assert answer["score"] == pytest.approx(1.5)  # uniform over 4 levels
    assert answer["legend"] == {"0": "low", "1": "medium", "2": "high", "3": "critical"}
    assert list(answer["probabilities"]) == ["0", "1", "2", "3"]
    assert answer["probabilities"]["0"] == pytest.approx(0.25)
    assert answer["confidence"] == 0.0


def test_noul_response_shape(base_url):
    status, body = post(
        base_url,
        systemone_url(),
        {
            "state": "The sky.",
            "questions": {
                "blue": {
                    "type": "noul",
                    "instructions": "The sky is blue.",
                }
            },
        },
    )
    assert status == 200
    answer = body["answers"]["blue"]
    assert set(answer) == {"type", "noul", "x_confidence", "x_temperature"}
    assert answer["type"] == "noul"
    assert answer["noul"] == pytest.approx(0.5)  # uniform stub
    assert answer["x_confidence"] == pytest.approx(0.0)


def test_list_form_questions(base_url):
    status, body = post(
        base_url,
        systemone_url(),
        {
            "state": "x",
            "questions": [
                {
                    "id": "a",
                    "type": "noul",
                    "instructions": "prop",
                },
                {
                    "qid": "b",
                    "type": "score",
                    "instructions": "rate",
                    "levels": ["lo", "hi"],
                },
            ],
        },
    )
    assert status == 200
    assert list(body["answers"]) == ["a", "b"]


def test_list_form_requires_id(base_url):
    status, body = post(
        base_url,
        systemone_url(),
        {"state": "x", "questions": [{"type": "noul", "instructions": "p"}]},
    )
    assert_422(status, body, ["body", "questions"])


def test_duplicate_question_ids_422(base_url):
    status, body = post(
        base_url,
        systemone_url(),
        {
            "state": "x",
            "questions": [
                {"id": "a", "type": "noul", "instructions": "p"},
                {"id": "a", "type": "noul", "instructions": "q"},
            ],
        },
    )
    assert_422(status, body, ["body", "questions"])


# ---------------------------------------------------------------------------
# Confidence formula: (N * pmax - 1) / (N - 1)
# ---------------------------------------------------------------------------


def test_confidence_formula_choice():
    backend = FixedBackend(fn=lambda p, n: [3.0, 0.0])
    with live_server(backend=backend) as url:
        status, body = post(url, systemone_url(), simple_payload())
        assert status == 200
        probs = body["answers"]["deploy"]["probabilities"]
        pmax = max(probs.values())
        expected = (2 * pmax - 1) / 1
        assert body["answers"]["deploy"]["confidence"] == pytest.approx(expected)


def test_confidence_uniform_is_zero(base_url):
    status, body = post(base_url, systemone_url(), simple_payload())
    assert body["answers"]["deploy"]["confidence"] == pytest.approx(0.0)


def test_confidence_point_mass_is_one():
    backend = FixedBackend(fn=lambda p, n: [30.0, 0.0, 0.0])
    with live_server(backend=backend) as url:
        status, body = post(
            url,
            systemone_url(),
            {
                "state": "x",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "pick",
                        "options": ["a", "b", "c"],
                    }
                },
            },
        )
        assert status == 200
        assert body["answers"]["q"]["confidence"] == pytest.approx(1.0)


def test_noul_confidence_binary():
    backend = FixedBackend(fn=lambda p, n: [0.0, 3.0])  # [not-true, true]
    with live_server(backend=backend) as url:
        status, body = post(
            url,
            systemone_url(),
            {
                "state": "x",
                "questions": {"q": {"type": "noul", "instructions": "prop"}},
            },
        )
        assert status == 200
        answer = body["answers"]["q"]
        p = answer["noul"]
        assert p > 0.9
        # binary confidence: 2 * pmax - 1
        assert answer["x_confidence"] == pytest.approx(2 * p - 1)


# ---------------------------------------------------------------------------
# Multi-question batching and per-question isolation
# ---------------------------------------------------------------------------


def test_full_lm_head_rows_trimmed():
    """Backends may return full 26-letter rows (like the torch readout):
    decoding must trim to valid letters, incl. the 2-logit noul readout."""

    backend = FixedBackend(fn=lambda p, n: [0.1 * i for i in range(26)])
    with live_server(backend=backend) as url:
        status, body = post(
            url,
            systemone_url(),
            {
                "state": "x",
                "questions": {
                    "q": {"type": "noul", "instructions": "prop"},
                },
            },
        )
        assert status == 200
        assert 0.0 < body["answers"]["q"]["noul"] < 1.0


def test_multi_question_batching():
    backend = FixedBackend()
    with live_server(backend=backend) as url:
        status, body = post(
            url,
            systemone_url(),
            {
                "state": {"build": "red", "friday": True},
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Route the ticket.",
                        "options": ["oncall", "queue"],
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "Rate urgency.",
                        "levels": ["low", "high"],
                    },
                    "is_flaky": {
                        "type": "noul",
                        "instructions": "The failure is a flake.",
                    },
                },
            },
        )
        assert status == 200
        assert set(body["answers"]) == {"route", "urgency", "is_flaky"}
        # One prompt per question (per-question isolation), one batch call.
        assert len(backend.calls) == 3
        prompt_texts = [c[0] for c in backend.calls]
        # each prompt mentions the state and only its own question
        assert "oncall" in prompt_texts[0]
        assert "is a flake" not in prompt_texts[0]
        assert "low" in prompt_texts[1]
        assert "oncall" not in prompt_texts[1]
        assert "Answer 1: (" in prompt_texts[2]
        for prompt in prompt_texts:
            assert "route" not in prompt  # question ids never appear
        # usage tokens sum the per-prompt token counts
        assert body["usage"]["input_tokens"] == sum(
            len(p.split()) for p in prompt_texts
        )


# ---------------------------------------------------------------------------
# Temperature application
# ---------------------------------------------------------------------------


def test_temperature_flattens_distribution(tmp_path):
    backend = FixedBackend(fn=lambda p, n: [2.0, 0.0])
    cal_path = tmp_path / "calib.json"
    cal_path.write_text('{"per_primitive": {"choice": 2.0}}')
    from deem_server import Calibration

    with live_server(
        backend=backend, calibration=Calibration.from_file(cal_path)
    ) as url:
        status, body = post(url, systemone_url(), simple_payload())
        assert status == 200
        answer = body["answers"]["deploy"]
        assert answer["x_temperature"] == 2.0
        # T=2 halves the logit gap: softmax([1, 0])
        expected = math.exp(1.0) / (math.exp(1.0) + 1.0)
        assert answer["probabilities"]["deploy"] == pytest.approx(expected)


def test_per_dataset_temperature_wins():
    from deem_server import Calibration

    cal = Calibration.from_dict(
        {
            "per_primitive": {"choice": 1.0},
            "per_dataset": {"ag_news": 4.0},
        }
    )
    backend = FixedBackend(fn=lambda p, n: [2.0, 0.0, 0.0])
    with live_server(backend=backend, calibration=cal) as url:
        status, body = post(
            url,
            systemone_url(),
            {
                "state": "x",
                "questions": {
                    "with_ds": {
                        "type": "choice",
                        "instructions": "pick",
                        "options": ["a", "b", "c"],
                        "dataset": "ag_news",
                    },
                    "without_ds": {
                        "type": "choice",
                        "instructions": "pick",
                        "options": ["a", "b", "c"],
                    },
                },
            },
        )
        assert status == 200
        assert body["answers"]["with_ds"]["x_temperature"] == 4.0
        assert body["answers"]["without_ds"]["x_temperature"] == 1.0
        assert (
            body["answers"]["with_ds"]["confidence"]
            < body["answers"]["without_ds"]["confidence"]
        )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_404_unknown_path(base_url):
    status, body = get(base_url, "/nope")
    assert status == 404
    assert body["detail"]["error_type"] == "not_found"
    assert body["detail"]["message"]


def test_405_get_on_systemone(base_url):
    status, body = get(base_url, "/v1/systemone")
    assert status == 405
    assert body["detail"]["error_type"] == "method_not_allowed"


def test_post_404(base_url):
    status, _ = post(base_url, "/v1/other", {"x": 1})
    assert status == 404


def test_backend_error_500():
    class ExplodingBackend(FixedBackend):
        def slot_logits(self, prompts, n_valids):
            from deem_server import BackendError

            raise BackendError("boom")

    with live_server(backend=ExplodingBackend()) as url:
        status, body = post(url, systemone_url(), simple_payload())
        assert status == 500
        assert body["detail"] == {"error_type": "backend_error", "message": "boom"}


# ---------------------------------------------------------------------------
# Aux endpoints
# ---------------------------------------------------------------------------


def test_health(base_url):
    status, body = get(base_url, "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["model"] == "deem-test"
    assert body["backend"] == "fixed"


def test_models(base_url):
    status, body = get(base_url, "/v1/models")
    assert status == 200
    assert body == {
        "models": [
            {
                "name": "deem-test",
                "description": "Deem typed decision model",
                "release_date": "",
            }
        ]
    }
