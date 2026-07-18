# DeepSeekV4/Decode

Orchestration of recurrent inference: initialization, prefill, single-token
forward, KV management, output head and diagnostics.

## Structure

- [`Execution/`](Execution/README.md): decoder state and per-layer path.
- [`Prefill/`](Prefill/README.md): layer-major ingestion of multiple tokens.
- [`Generation/`](Generation/README.md): embedding/output head and generation.
- [`State/`](State/README.md): reusable scratch buffers.
- [`KV/`](KV/README.md): snapshot and restore of the recurrent state.
- [`Attention/`](Attention/README.md): CPU top-k selection of the indexer.
- [`Cache/`](Cache/README.md): expert LRU cache and statistics.
- [`Diagnostics/`](Diagnostics/README.md): timing profile and I/O.
- [`Reference/`](Reference/README.md): reference implementation for parity.

## Flow

The factory prepares runtime, weights and caches. Prefill traverses the prompt
in layer-major chunks, updating the same state used by decode. Generation then
runs one forward per token: embedding -> layers -> attention/KV -> router and
FFN -> output head. Snapshots allow suspending and resuming the flow.

The raw KV is a circular/linear window bounded by `nSWA`; older context
survives in the compressed NSA rows. The runtime options are in the
[Configuration Reference](../../../../../README.md#configuration-reference).

## Modification rules

Prefill and decode must preserve the same recurrent semantics. Every
asynchronous path must define ownership, wait point and cancellation. A change
to the KV layout requires a coordinated update of snapshots, checkpoints and
tests.
