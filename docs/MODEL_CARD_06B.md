---
library_name: deem
license: apache-2.0
base_model: Qwen/Qwen3-0.6B-Base
tags:
- decision-model
- cpu-inference
- rust
- edge
- routing
metrics:
- policy_holdout
- macro_accuracy
---

# Deem 0.6B (v1)

**Decisions, anywhere — smaller.** The Deem decision stack on
**CPU** at 0.6B parameters: typed, calibrated choices, scores, and
yes/no decisions with abstention, served by our own Rust runtime.
No GPU required.

## Why 0.6B?

Qwen3-0.6B-Base has dense attention where its Qwen3.5 successor
hybridizes with linear attention. Through the Deem letter-slot
fine-tune that matters: **0.6B beats 0.8B** on the standard suite
(0.7533 vs 0.7455 macro) and on JevBench hard (40.5 vs 33.3) —
with 25% fewer parameters.

## Benchmarks

- **94.5%** long-policy hold-out accuracy
- Standard-suite macro **0.7533** (0.7622 calibrated)
- JevBench public **97.9 / 73.6 / 40.5** (easy / original / hard)
- Near-ceiling exact-law reasoning: counting 0.988, grid 0.992,
  zero-count 1.000 — best exact-laws Brier on the
  [Tare board](https://github.com/Libertai/deem) (0.0289)
- Temporal probes match the 9B: sharp 30/31-day window cutoff,
  consistency sums 1.000
- **65ms** short-form decisions, **1.9s** ~3k-token states
  (int8 path, busy desktop CPU)
- **~0.7GB** resident (int8 path)

## The runtime

A single static Rust binary. No Python, no C++ dependencies at
runtime.

- Hand-written **AVX-512 kernels** — bf16 (`vdpbf16ps`) and int8
  (`vpdpbusd`) GEMM lanes
- **Parity-gated against torch:** max letter-logit diff 0.080
  bf16 / 0.125 int8 (gate 0.35)
- **Wire-compatible `/v1/systemone`** — drop-in for the TypeSafe SDK

```bash
git clone https://github.com/Libertai/deem && cd deem/rust
cargo build --release
DEEM_CHECKPOINT=LibertAIDAI/deem-0.6-v1 ./target/release/deem-server
```

Runs where GPUs don't: CI runners, edge boxes, laptops, serverless
micro-VMs.

## How it's built

Qwen3-0.6B-Base (Apache-2.0, license tag + LICENSE file verified
on HF 2026-09-27), full fine-tune on the Deem
ground-truth-verified mixture (124,765 rows: anchors + exact laws
+ policy + working-time + verification traces), then the
temporal-window delta (6k windowgen + 15k replay, lr 1e-5).
Apache-2.0 recipe.

## License

Apache-2.0. All benchmarks reproducible from the
[release artifacts](https://github.com/Libertai/deem).
