# Four native Swift model decoders

The `ternary-bonsai-2-support` reference is pinned at
`acf5c16bb6a01f8c4f027a0db265ae270d8907ab`. Normal inference stays entirely in
Swift and Metal: no C engine, external inference process or static C library.

| Model | GGUF architecture | Native text path |
|---|---|---|
| Ternary Bonsai 2 27B | `qwen35`, validated Prism schema | PQ2/PTQ, folded Hadamard, GDN and grouped-query attention |
| Qwen3.8 Flash Next | `qwen4exp` | HC, GDN, sparse attention, MoE and original BF16 n-grams on SSD |
| DeepSeek V4.1 Flash | `deepseek41` | Delayed HC, compressed attention, shared index selections, MoE and Engram rows on SSD |
| GLM5.3 Flash | `glm5-next` | KDA recurrence, DSA sparse attention, HC and MoE |

Each decoder validates metadata and tensor shapes before inference. An ordinary
Qwen3.5 GGUF is not automatically a Bonsai checkpoint. DeepSeek V4.1 is separate
from the existing V4 graph. GLM5.3 Flash is separate from `glm-dsa`.

The app's chat and local API share `SwiftModelChatService`, a serial Swift actor
on a dedicated GCD executor. It applies native chat framing and tool grammars,
streams reasoning and visible text, reuses exact cached prefixes, and resets
partial recurrent/KV state after cancellation or a failed evaluation. These
backends expose text generation, reasoning and tools. Their image encoders,
MTP/speculative decoding, disk KV persistence, distributed execution and the
older DeepSeek-specific tuning controls are not enabled.

## Kernels and memory

Canonical shaders live in `metal/bonsai`, `metal/qwen38`, `metal/deepseek41` and
`metal/glm53`. Run `python3 scripts/embed_backend_kernels.py` to regenerate their
Swift strings, or `make embed-kernels` for all families. The new shaders are
compiled only for the selected backend. Bonsai preserves its independent safe
Metal math mode; the other backends retain the reference runtime defaults.

Prefill processes rows in layer-major batches and uses the architecture's
packed or dense matrix kernels. Routed experts are bounded to active sets;
Engram and Qwen's original n-gram tables are read through SSD descriptors, never
registered as whole GPU buffers. Very large GGUF file sizes remain very large
disk requirements. File size is not an estimate of resident memory or proof of
acceptable performance on a 32 GB machine.

## Validation and limits

Validated on 2026-09-21 with Apple M1 Max, macOS 27.0 and Xcode 27.0 / Swift 6.4:
the release app and CLI build, 56 focused tests pass, and the complete suite
reports 1,005 tests: 922 passed, 83 skipped, zero failures. Skips require absent
real checkpoints, optional benchmark inputs or legacy kernel-directory fixtures.
The Qwen synthetic GPU graph was explicitly enabled. The regenerated Xcode
project passes plist validation, and shader embedding is deterministic without
changing the existing DeepSeek kernel bundle. No UI automation was performed.

Model-free tests compare packed coefficients, Hadamard transforms, recurrent
state, BF16/FP8/FP4 rounding, pooling, sparse selection and projection kernels
against CPU/reference fixtures. Qwen includes a small synthetic full graph
that compares batched and incremental execution, reset and cancellation.
Tokenizer golden data comes from the pinned C pretokenizer. Chat-service tests
exercise prefix reuse, history replacement, error recovery, UTF-8 streaming,
tool delimiters, concurrent request rejection and cancellation through the same
`ChatBackend` interface used by the UI and server.

Example focused validation on macOS with full Xcode selected:

```sh
DS4_TEST_QWEN38_GPU=1 swift test -c release \
  --filter 'Bonsai|Qwen38|QwenTokenizerGolden|GLM53|DeepSeek41|SwiftModelChatService|NativeSwiftModelIntegration|NativeProtocolReview|ModelAssembly|DownloadAssemblyState'
```

Synthetic numerical checks do not establish full-checkpoint logits parity,
answer quality or end-to-end throughput. No new real checkpoint was downloaded
to validate this port. The new decoders remain experimental until those
measurements are available. To run an already downloaded model locally:

```sh
DS4_DEMO_CONTEXT=4096 swift run -c release DS4Demo /path/model.gguf 32 'Ciao'
```

The catalog uses immutable Hugging Face revisions, expected sizes and SHA-256
digests. Artifact provenance is recorded in
[FOUR-MODEL-INTEGRATION-NOTES.md](FOUR-MODEL-INTEGRATION-NOTES.md); code attribution
is in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Downloads

The model picker includes Bonsai PQ2/PTQ, Qwen Q2/Q4, DeepSeek V4.1 Q2/Q4 and
GLM5.3 Flash Q2/Q4_K. DeepSeek V4.1 Q4 downloads two byte fragments, then
assembles and verifies one GGUF before it becomes selectable. Cancelled merges
retain a resumable partial file; source fragments remain on disk. Its peak disk
requirement is about 1.04 TB plus the free-space reserve, shown before download.

The catalog also offers the five matching encoder variants, Bonsai F16, GLM
Flash FP8 and full GLM5.3 IQ2 as explicitly download-only entries. These do not
enable image understanding or unsupported weight formats, and they never
replace the selected language model. Download tests use tiny local fragments
to cover checksums, cancellation, resume, atomic publication and model selection
after the original fragments are removed.
