# Deem serving lane

`/v1/systemone`-compatible HTTP server + MCP stdio server for Deem typed
decision models (SPEC §2). **Zero-dependency**: stdlib `http.server` +
threading. FastAPI is *not* installed in either project virtualenv (checked
`.venv` and `.venv-sft` on 2026-09-21) and the core readout
(`src/deem/format.py`) is stdlib-only, so the HTTP layer stays stdlib too.
The only heavy dependency — torch + transformers — is imported lazily by the
real-checkpoint backend.

| File | What |
|---|---|
| `deem_server.py` | HTTP server: `POST /v1/systemone`, `GET /v1/models`, `GET /health` |
| `deem_mcp.py` | MCP server (JSON-RPC 2.0 over stdio, no MCP SDK): `classify` / `score` / `check` |
| `tests/` | validation, TypeSafe criteria + prompt rendering, response shape, confidence formula, batching, temperature, error paths, MCP dispatch + stdio round-trip |

## Quickstart

```bash
# Deterministic stub backend (uniform logits) — any Python ≥3.10:
.venv/bin/python serve/deem_server.py

# Real checkpoint (v4 SFT, Qwen3-1.7B) + published calibration temperatures:
DEEM_CHECKPOINT=scripts/sft/checkpoints/v4_17b \
DEEM_CALIBRATION=scripts/sft/temperature_v4.json \
DEEM_CALIBRATION_KEY=v4 \
    .venv-sft/bin/python serve/deem_server.py --model-id deem-1.5
```

No checkpoint configured? The server logs it and serves uniform-logit stub
answers (choice/score confidence 0, noul 0.5) — useful for wiring up
clients before weights exist.

## Endpoint reference

### POST /v1/systemone

One state, many typed questions, all answered against that state in a
single request (one isolated forward pass per question, batched).

```bash
curl http://127.0.0.1:8300/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "Deploy box prod-1: build #4021 green, 3 unit tests flaky, Friday 16:55.",
    "questions": {
      "ship_it": {"type": "choice", "instructions": "What should we do?",
                   "criteria": {"deploy": "Ship build #4021 now",
                                "hold": "Wait until Monday",
                                "rollback": null}},
      "risk":    {"type": "score", "instructions": "Rate the release risk.",
                   "criteria": ["low", "medium", "high"]},
      "flaky":   {"type": "noul", "instructions": "Are the failures flakes?",
                   "criteria": {"true": "Tests fail intermittently",
                                "false": "A real regression"}}
    }
  }'
```

Response:

```json
{
  "model": "deem-1.5",
  "answers": {
    "ship_it": {"type": "choice", "choice": "deploy",
                "probabilities": {"deploy": 0.86, "hold": 0.11, "rollback": 0.03},
                "confidence": 0.79, "x_temperature": 0.9888},
    "risk":    {"type": "score", "score": 0.95,
                "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0.22, "1": 0.61, "2": 0.17},
                "confidence": 0.42, "x_temperature": 3.1187},
    "flaky":   {"type": "noul", "noul": 0.83, "x_confidence": 0.66,
                "x_temperature": 4.3325}
  },
  "usage": {"input_tokens": 84, "output_tokens": 0}
}
```

Contract (TypeSafe `/v1/systemone`):

- **state** — any JSON value (string, object, …); serialized canonically
  and hardened against `</state>` injection (see `src/deem/format.py`).
- **questions** — object keyed by question id. Per question:
  - `instructions`: string (verbatim) or object/array (rendered as
    canonical JSON); optional, missing → empty.
  - `choice`: `criteria` maps each option (2–255 unique, non-empty, no
    newlines; order kept) to a description or `null`. Descriptions are
    rendered into the prompt as `(A) option: description`.
  - `score`: `criteria` is the list of 2–10 levels, low → high.
  - `noul`: optional `criteria` `{"true": ..., "false": ...}` (either side
    optional), rendered as `True if: ...` / `False if: ...` lines.
  - Non-string descriptions / levels / noul criteria render as canonical
    JSON; newlines inside them become spaces.
  - optional `dataset`: calibration-temperature key for this question.
- Legacy aliases still accepted: `options` / `levels` string lists instead
  of `criteria`, and `questions` as a list of objects with an `id` field.
  A top-level `model` is accepted and ignored; optional top-level
  `dataset` applies to all questions lacking their own. At most
  `DEEM_MAX_QUESTIONS` (64) questions per request.
- Answers: choice `{choice, probabilities, confidence}`; score
  `{score, legend, probabilities, confidence}` where `score` is the
  expected 0-based level index and `probabilities` / `legend` are keyed
  `"0"`..`"n-1"`; noul `{noul}`. Non-spec fields carry an `x_` prefix
  (`x_temperature`, noul `x_confidence`). Usage is
  `{"input_tokens", "output_tokens": 0}`.
- `confidence` is derived from the calibrated distribution:
  `(N·pmax − 1)/(N−1)` (0 for uniform, 1 for point mass); for noul it is
  the same formula on the binary {p, 1−p} distribution, i.e. `2·pmax − 1`.
- The `torch` backend reads single-token letters only, so it supports at
  most 26 options per question (TypeSafe allows 255; more than 26 is a
  422 on `criteria`). Multi-token letter readout `AA`, `AB`, … is a
  training-lane follow-up (see `format.py`).

Errors: validation failures are **422** with a FastAPI-style body

```json
{"detail": [{"loc": ["body", "questions", "risk", "criteria"],
             "msg": "needs 2-10 entries, got 1", "type": "value_error"}]}
```

(invalid JSON: `"type": "json_invalid"`, `"loc": ["body"]`). Everything
else — 404, 405, 413 (body over 8 MB), 500 (backend) — is
`{"detail": {"error_type": "...", "message": "..."}}`.

### GET /v1/models

```json
{"models": [{"name": "deem-1.5", "description": "Deem typed decision model",
             "release_date": ""}]}
```

### GET /health

`{"status": "ok", "model": "deem-1.5", "backend": "torch"}`

## Calibration (temperature scaling)

Per-dataset / per-primitive temperatures are loaded from a JSON file
(`DEEM_CALIBRATION`, optional). The emitted
`scripts/sft/temperature_v4.json` shape is accepted directly — it nests
`per_primitive` / `per_dataset` under version keys, selected with
`DEEM_CALIBRATION_KEY`:

```bash
DEEM_CALIBRATION=scripts/sft/temperature_v4.json \
DEEM_CALIBRATION_KEY=v4 ...
```

Lookup order per question: `per_dataset[dataset]` → `per_primitive[type]`
→ 1.0. A flat `{"choice": 1.0, ...}` mapping also works. The applied
temperature is echoed in every answer.

## Latency expectations

Measured 2026-09-21 on a quiet RTX 5090 (no other GPU jobs running).
Checkpoint numbers go through `DeemCore` + `TorchBackend` directly —
the HTTP layer adds ~0.2 ms per request and is not the bottleneck. Each
figure is 100 requests after 10 warmup requests, single client, batch
size 4, v5 checkpoint (`scripts/sft/checkpoints/v5_17b`) +
`temperature_v5.json` calibration:

| Backend | Request | p50 | p99 |
|---|---|---|---|
| stub, over HTTP | 3 questions | 0.2 ms | 0.4 ms |
| v5 checkpoint (Qwen3-1.7B, bf16, RTX 5090) | 1 question | 11.3 ms | 12.9 ms |
| v5 checkpoint (Qwen3-1.7B, bf16, RTX 5090) | 3 questions | 12.5 ms | 13.3 ms |
| v4 checkpoint (Qwen3-1.7B, bf16, CPU) | 1 question | 137 ms | 155 ms |
| v4 checkpoint (Qwen3-1.7B, bf16, CPU) | 3 questions | 259 ms | 312 ms |

The GPU p50 is ~11 ms, not single-digit: the letter-slot readout is one
prefill-only batched forward pass per question (no decoding), and at
these short prompt lengths the cost is dominated by eager-mode kernel
launch overhead across the 28 transformer layers, not FLOPs — the GPU
clocks to full speed (3045 MHz) and sits at ~60% utilization during the
benchmark. The good news is in the 1-vs-3-question delta: questions in
one request share a single batched forward pass, so 3 questions cost
only ~1.1 ms more than 1 (batch size 4). Larger batches amortize
further.

Re-measure yourself:

```bash
.venv/bin/python - <<'EOF'
import sys, time, json, urllib.request
req = urllib.request.Request("http://127.0.0.1:8300/v1/systemone",
    data=json.dumps({"state": "x", "questions": {
        "q": {"type": "noul", "instructions": "prop"}}}).encode(),
    headers={"Content-Type": "application/json"})
urllib.request.urlopen(req).read()  # warmup
ts = []
for _ in range(200):
    t0 = time.perf_counter(); urllib.request.urlopen(req).read()
    ts.append((time.perf_counter() - t0) * 1e3)
ts.sort(); print(f"p50={ts[len(ts)//2]:.1f}ms p99={ts[-2]:.1f}ms")
EOF
```

## SDK compatibility

The official TypeSafe SDKs (checked: Python `typesafe-sdk` 0.7.2) work
against this server. The server ignores auth, but the SDKs refuse to
start without an API key, so pass any non-empty one plus the base URL:

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:8300
export TYPESAFE_API_KEY=local   # any non-empty value
```

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(
    api_key="local", base_url="http://127.0.0.1:8300", model="deem-1.5"
)
resp = client.system_one(
    "Help! My payouts have been failing for 3 days.",
    {
        "urgent": Noul(
            instructions="Is this urgent?",
            criteria={"true": "Time-sensitive", "false": "No urgency"},
        ),
        "team": Choice(
            instructions="Which team?",
            criteria={"technical": "Bugs, outages", "billing": None},
        ),
        "mood": Score(
            instructions="Customer mood?", criteria=["Calm", "Frustrated", "Very angry"]
        ),
    },
)
```

Put auth in front of the server at the proxy/gateway if you expose it.
Conformance: `uvx jevcompat test http://127.0.0.1:8300`.

## MCP server

```json
{
  "mcpServers": {
    "deem": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/serve/deem_mcp.py"],
      "env": {
        "DEEM_CHECKPOINT": "scripts/sft/checkpoints/v4_17b",
        "DEEM_CALIBRATION": "scripts/sft/temperature_v4.json",
        "DEEM_CALIBRATION_KEY": "v4"
      }
    }
  }
}
```

Tools: `classify(state, instructions, options)` → choice + probabilities
+ confidence; `score(state, instructions, levels)` → expected level index
(`score`) + `legend` + index-keyed distribution + confidence;
`check(state, instructions)` → `noul`, the probability the proposition is
true. Answers use the same TypeSafe shapes as the HTTP server. Use `--stub` for the deterministic stub. It shares
`DeemCore` with the HTTP server, so both lanes answer identically.

## Deployment (systemd)

```ini
# /etc/systemd/system/deem-serve.service
[Unit]
Description=Deem /v1/systemone server
After=network.target

[Service]
Type=simple
User=deem
WorkingDirectory=/opt/deem/newgen-model
Environment=DEEM_CHECKPOINT=scripts/sft/checkpoints/v4_17b
Environment=DEEM_CALIBRATION=scripts/sft/temperature_v4.json
Environment=DEEM_CALIBRATION_KEY=v4
Environment=DEEM_MODEL_ID=deem-1.5
Environment=DEEM_HOST=0.0.0.0
Environment=DEEM_PORT=8300
Environment=DEEM_DEVICE=cuda
Environment=DEEM_BATCH_SIZE=8
ExecStart=/opt/deem/newgen-model/.venv-sft/bin/python serve/deem_server.py
Restart=on-failure
# Hardening
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload && systemctl enable --now deem-serve
```

## Configuration reference

| Env var | Default | Meaning |
|---|---|---|
| `DEEM_CHECKPOINT` | *(unset → stub)* | HF checkpoint dir for the letter-slot backend |
| `DEEM_CALIBRATION` | *(unset → T=1)* | calibration JSON (see above) |
| `DEEM_CALIBRATION_KEY` | first entry | version key inside the calibration file |
| `DEEM_MODEL_ID` | `deem-1.5` | reported by the API |
| `DEEM_HOST` / `DEEM_PORT` | `127.0.0.1` / `8300` | bind address |
| `DEEM_DEVICE` | `auto` | `cuda`, `cpu` or `auto` |
| `DEEM_BATCH_SIZE` | `4` | forward-pass batch size |
| `DEEM_MAX_QUESTIONS` | `64` | per-request question cap |

## Tests

```bash
.venv/bin/pytest serve/tests/ -q
```

Covers request validation (option/level count limits, criteria shapes,
malformed JSON, unknown types, duplicate ids, 422 `detail` shape),
TypeSafe criteria rendering, response shape for all three primitives,
the derived-confidence formula, per-question isolation and batching,
per-dataset/per-primitive temperature application, all HTTP error paths
(404/405/413/422/500), MCP dispatch, and a stdio subprocess round-trip.
