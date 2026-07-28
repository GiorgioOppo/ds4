**English** | [Italiano](ARCHITETTURE-SUPPORTATE.it.md)

# Supported model architectures

This document defines the boundary between the shared parts of DwarfStar and
the backends specific to a model family. It is also the authoritative source on
support status: recognizing an architecture does not imply that its decoder is
already available.

## Current status

| Family | GGUF recognition | GUI catalog/download | Local inference | GUI selection and demo | Distribution | Notes |
|---|---:|---:|---:|---:|---:|---|
| DeepSeek V4 Flash | yes | yes | yes | yes | yes | Three complete quantizations are downloadable, selectable and runnable. |
| DeepSeek V4 Pro | yes | yes | yes, single-file Q2 | yes, single-file Q2 | yes, single-file Q2 | The distributed path is implemented and tested at the protocol level; the numerical validation with the real Pro GGUF is still pending. The Q4 split package remains download-only. |
| GLM 5.2 (`glm-dsa`) | yes | yes | yes, streaming engine | yes (chat, demo, server, benchmark, auto-tune) | no | End-to-end runnable: streaming engine with per-token expert gather, Q4 layer sidecar pack, incremental-KV server reuse and a prefix-keyed disk-KV store with token budget and eviction. Measured ~0.33 tok/s decode on the IQ2_XXS at 16 GB (see `architectures/glm-5.2/`). |
| Laguna S 2.1 (`laguna`) | yes | yes, download-only | no | no, explicit rejection | no | Staged port of the reference `laguna-s2.1` branch: geometry/metadata validation, tokenizer, native chat with interleaved reasoning and tagged tool calls, tensor schema of the published quant recipes and catalog entries are in place; inference is refused behind `LagunaRuntimeGate` until the Metal decoder is ported and passes logits parity (see `PORTING-GAPS.md`). |
| Qwen | yes | no | no | no, explicit rejection | no | Structure is in place; decoder and templates are not yet implemented. |
| Unknown | yes | no | no | no, explicit rejection | no | The GGUF identifier is reported without attempting a DeepSeek load. |

Selection uses the GGUF's `general.architecture`, normalized into a stable
identifier. A Qwen or GLM file must never reach the DeepSeek validator and
produce a misleading error about `deepseek4.*` metadata. GLM already has its
own frontend, but that does not change runtime availability.

## GUI catalog and runtime support

`ModelCatalogRegistry` in `DS4Engine/ModelManagement/Catalog` is the single
cross-family source for the remote models shown by the GUI. It maintains the
family-specific catalogs `DeepSeekV4ModelCatalog` and `GLM52ModelCatalog`. The
registry also describes artifacts that this build can download but not run: the
presence of a row therefore does not equal decoder support.

The current entries are:

- DeepSeek V4 Flash Q2 imatrix, mixed Q2/Q4 imatrix and Q4 imatrix: one
  complete GGUF each, `runnable` and selectable;
- DeepSeek V4 Pro Q2: one complete GGUF, `runnable` and selectable for the
  local runtime;
- DeepSeek V4 Pro Q4: `downloadOnly` package of two shards; the shards are not
  presented as independent local models;
- GLM 5.2 IQ2_XXS, Q2_K and Q4_K: three alternative monolithic GGUFs from the
  `antirez/glm-5.2-gguf` repository, selectable and runnable through the GLM
  backend;
- Laguna S 2.1 Q4_K_M (official Poolside file, revision-pinned) and the mixed
  RoutedQ2_K/Last27Q3_K requant: `downloadOnly` behind the Laguna runtime
  gate; they become selectable only when the decoder port enables the gate;
- MTP and the Laguna DFlash Q8_0 draft: separate accessories, excluded from
  the main model catalog and from GUI selection.

The GUI's automatic scan filters on the filenames the catalog declares
selectable: the three Flash entries, the single Pro Q2 and the three GLM
GGUFs. **Browse** allows an external file, but first
performs the metadata-only inspection and backend selection: a Pro Q4 shard,
MTP, Laguna (while its gate is off), Qwen and unknown architectures cannot
silently replace the active model, while a GLM 5.2 GGUF routes to its own
concrete backend.

Pro Q2 support covers both the local decoder and distributed pipeline and
expert-parallelism. The distributed geometry is covered by protocol and
partitioning tests; the numerical/multi-Mac campaign with the real Pro GGUF,
which was not available during this implementation, remains explicitly pending.

## Code boundaries

The following remain common across architectures:

- GGUF reading and mapping;
- sampling and token representation;
- Metal runtime, device management, command queues and pipelines;
- application APIs, benchmarks, server, agents, tools and MCP;
- file transport and the persistent checkpoint envelope;
- GUI state and persistence.

The following remain in the backend that owns them:

- model shape and metadata validation;
- tokenizer, special tokens and conversational template;
- tensor names, layout and loading;
- decoder, prefill, KV payload and numerical diagnostics;
- MoE routing, NSA, Hyper-Connection, MTP and expert cache;
- distributed capabilities tied to the model's geometry.

This separation avoids conditional options inside the per-layer loop. Backend
selection happens once when the model is opened; the hot path keeps using
concrete types.

## Selection contract

Loading follows this sequence:

1. open the GGUF and read the architecture identifier;
2. build a neutral description of the model and its capabilities;
3. select the backend registered for that family;
4. run family-specific validation and build the concrete decoder;
5. publish the description to demo, GUI and diagnostics.

An architecture that is recognized but not implemented produces an error
distinct from a corrupt file or from an unsupported profile of the same family.
This is the expected behavior for Qwen during its preparatory phase. A valid DeepSeek V4 Pro Q2, on the other hand, goes through
the same concrete backend as Flash, but with a `DSV4RuntimeGeometry` built from
the metadata: 61 layers, 7168 channels, 128 heads, 384 experts, top-1024
indexer and router scale 2.5 are not replaced by the Flash constants.

## Compatibility

The restructuring does not change:

- the `DS4Core`, `DS4Metal`, `DS4Engine`, `DS4Demo` and `DwarfStar` targets;
- the app name, bundle identifiers and Application Support folders;
- the existing `UserDefaults` keys and `DS4_*` environment variables;
- the historical public types kept as aliases or façades;
- the format and reading of DeepSeek checkpoints already created.

New checkpoints will have to include the model's architecture and fingerprint.
A future Qwen KV payload will get its own type/version and will not reuse the
DeepSeek payload while implicitly changing its meaning.

## Requirements before the Qwen backend

Before declaring Qwen usable, the following are needed for one precise GGUF
variant:

1. inventory of the real metadata and tensor layouts;
2. tokenizer and chat template with golden tests, including reasoning, tools
   and stops;
3. separate Metal decoder with numerical tests for prefill and single token;
4. KV and context management with its own persistent version;
5. capability-based settings profile, without showing DeepSeek knobs;
6. smoke test with a small GGUF and correctness benchmark;
7. only afterwards, explicit design of Qwen distribution.

No generic "Qwen" is chosen in advance: Qwen2, Qwen2-MoE, Qwen3 and their
variants may have different contracts. Implementation will start from a
declared GGUF file and variant, not from the commercial name alone.

## GLM 5.2 backend status

The GLM catalog pins repository, revision, filename, byte count and SHA-256.
The requirements that gated the `runnable` state have been met: `glm-dsa` is
registered, the complete MLA/DSA graph, routed/shared MoE, weight loader and
streaming, runtime KV and logits output are implemented, and prefill/decode
have been validated numerically against the real IQ2_XXS GGUF
(`GLM52RuntimeGate.enabled` is on, which flips the catalog entries to
`runnable`). MTP (the nextn block) is not part of the autoregressive path.
GLM uses its own `DS4_GLM_*` knob namespace; DeepSeek knobs are not
implicitly reused. Engine details and measured optimizations are in the
[GLM 5.2 README](architectures/glm-5.2/README.md).

## Laguna S 2.1 backend status

Laguna follows the same staged path GLM followed. Implemented and unit-tested
today, entirely without model files: `laguna` registration and detection, the
exact `DS4_SHAPE_LAGUNA_S21` geometry validation (48 blocks, GQA with 8 KV
heads, the per-layer 48/72 query-head alternation and its sliding-window rule,
YaRN with an independent SWA frequency base), the Laguna BPE pre-tokenizer
(newline-run pre-split plus single-digit GLM4-shape groups), the native chat
template with interleaved reasoning and tagged tool calls, the reference
sampling defaults (temperature 0.7, top-k 20, top-p 0.95, min-p 0.05), the
tensor schema of the two published recipes plus the mixed Q2_K/Q3_K file, and
the download catalog. What is missing is the decoder: `metal/laguna.metal`
and its driver, plus the optional DFlash speculative companion, are tracked as
a turnkey plan in [`PORTING-GAPS.md`](PORTING-GAPS.md).
`LagunaRuntimeGate.enabled` stays off — and every Laguna GGUF is refused with
a distinct error — until that port passes end-to-end logits parity on real
weights.

## Per-family documents

- [`architectures/deepseek-v4/README.md`](architectures/deepseek-v4/README.md)
  describes the operational backend and its constraints.
- [`architectures/qwen/README.md`](architectures/qwen/README.md) describes the
  groundwork and what remains to be implemented.
- [`architectures/glm-5.2/README.md`](architectures/glm-5.2/README.md)
  describes the verified contract, the streaming engine and the measured
  optimizations of the GLM backend.
- [`architectures/laguna-s-2.1/README.md`](architectures/laguna-s-2.1/README.md)
  describes the staged Laguna frontend and what the decoder port still
  requires.
- [`STRUTTURA-PROGETTO.md`](STRUTTURA-PROGETTO.md) indicates where to place
  files.
