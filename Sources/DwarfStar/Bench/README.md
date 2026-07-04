# DwarfStar/Bench

- **`BenchController.swift`** runs the native benchmark. It measures prefill and
  generation throughput in tokens/second at increasing context sizes. The engine
  can be **Local** (the loaded shared in-process engine, gated so chat is idle)
  or **Distributed** (reusing the connected coordinator).
- **`BenchView.swift`** renders the engine selector, context boundaries,
  throughput chart with Swift Charts, and running-engine indicator.

## Sample results

DeepSeek V4 Flash (43 layer, IQ2_XXS/Q2_K experts) on an Apple M1 Pro, shared
in-process engine, context window configured at 104k (KV ~1.35 GB), 64
generated tokens per point:

```text
Running on the shared engine (no second model copy)...
  ctx  64 · prefill 4.6 t/s · gen 1.30 t/s
  ctx 128 · prefill 5.3 t/s · gen 1.28 t/s
  ctx 192 · prefill 5.7 t/s · gen 1.28 t/s
  ctx 256 · prefill 6.1 t/s · gen 1.28 t/s
```

| Context tokens | Prefill t/s | Generation t/s |
|---:|---:|---:|
| 64  | 4.6 | 1.30 |
| 128 | 5.3 | 1.28 |
| 192 | 5.7 | 1.28 |
| 256 | 6.1 | 1.28 |

```
 t/s
 6 |                        ● prefill
 5 |            ●     ●
 4 |    ●
 3 |
 2 |
 1 |    ●-------●-----●-----● generation
 0 +----+------+-----+-----+---
      64     128   192   256   context tokens
```

How to read it: **prefill throughput RISES with the prompt length** — the
layer-major batching amortizes the per-chunk fixed costs (dense reloads,
expert-union gathers, command-buffer syncs) over more tokens, so longer
prompts prefill *faster* per token (measured up to ~8 t/s at 3k tokens on the
same machine with the demo CLI). **Generation stays flat** at these context
sizes — the decode is SSD-gather-bound (one token at a time, ~0.6 GB of
expert reads per token), so the context length barely matters until the
attention cost catches up at multi-thousand-token contexts.

Note on absolute numbers: generation here reads ~1.3 t/s because this run was
taken with the context window configured at 104k — the pre-allocated KV/
compressor buffers at that setting pressure a 16 GB machine. The same engine
with the context at 4–8k measures ~2.5 t/s in the decode profile. Prefill and
generation also aren't directly comparable to the demo CLI numbers unless the
`DS4 engine:` knob lines match.
