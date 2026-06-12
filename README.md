# DwarfStar — DeepSeek-V4 on macOS

Native Swift / SwiftUI app for **DeepSeek-V4-Flash (284B MoE)** on Apple Silicon,
with a **pure-Swift Metal inference engine** (a faithful port of the upstream
`ds4.c` / `ds4_metal.m`). No C engine, no prebuilt static lib, no external links.
The 2-bit GGUF runs on a 16 GB MacBook by streaming weights from SSD.

> 📖 **Documentazione dettagliata (IT):** [`docs/DOCUMENTAZIONE.md`](docs/DOCUMENTAZIONE.md)
> (uso, GUI, tool, agenti, server, distribuito) · [`docs/ARCHITETTURA-MOTORE.md`](docs/ARCHITETTURA-MOTORE.md)
> (interni del motore: tokenizer, decoder, MoE, NSA, streaming, cache esperti).

## Architecture

```
DwarfStar (SwiftUI)            ← chat · agenti · progetti · tuning · server ·
        │                        distribuito · benchmark · diagnostica
   DS4Engine (Swift)           ← InferenceService (actor): prompt → event stream;
        │                        tools (DSML) · agenti · ProjectCache ·
        │                        distributed runtime (worker/coordinator)
   DS4Core + DS4Metal (Swift)  ← pure-Swift engine: GGUF mmap, tokenizer, sampler,
        │                        Metal runtime + decode graph, expert slot-cache
   metal/*.metal               ← kernels, embedded in the binary at build time
```

The engine is a faithful Swift reimplementation; correctness is the project's
#1 rule, validated by the tests in `Tests/DS4CoreTests/`.

### Engine key facts

- **SSD streaming is the only path.** Non-routed weights are no-copy `mmap`
  views (resident via the OS page cache, evictable); per token only the
  **6/256 routed experts** of the current layer are gathered. The model never
  needs to fit in RAM.
- **Metal kernels embedded in the binary** (`make embed-kernels` regenerates
  `Sources/DS4Metal/Runtime/KernelSources.swift` from `metal/*.metal`). No
  on-disk `metal/` folder is needed at runtime. Fused MoE kernels
  (pair-SwiGLU / down-sum6) cover q4_K, q2_K, iq2_xxs experts.
- **Expert slot-cache + usage imatrix.** An optional per-layer LRU pool keeps
  hot experts resident on GPU; routing-frequency statistics are persisted
  per model **and per agent** and pre-warm the cache (Tuning tab).
- **Multi-turn KV reuse.** The conversation is append-only (`committedIds`):
  each turn prefills only the new suffix; an interrupted generation rebuilds
  the KV from the exact committed ids (the NSA compressor is recurrent and
  cannot rewind).
- **Tool calling (DSML).** The DeepSeek-V4 native format: an XML scheme on the
  `｜DSML｜` control token, rendered exactly like the GGUF's `chat_template`
  (compact or full declaration). Tool results return as
  `<tool_result>…</tool_result>` inside a user turn.
- **Layer-major prefill** in chunks (512): each layer's weights are loaded once
  per chunk and applied to all its tokens; the routed-FFN phase gathers the
  **union** of the chunk's experts once instead of 6 per token.

## The app

| Tab | What it does |
|---|---|
| **Chat** | streaming chat: reasoning collassabile, tool-call live (markup grezzo → card), risultati tool, riuso KV multi-turno, import progetto |
| **Agenti** | ruoli con prompt/icone/tool per agente, editor completo, export/import JSON, profilo di uso esperti per agente |
| **Progetto** | libreria di progetti (cartelle indicizzate, bookmark sandbox); i tool `project_list/read/search` li esplorano senza toccare la memoria chat |
| **Tuning** | slot della cache esperti, hit-rate, concentrazione del routing per layer (la "usage imatrix") |
| **Server** | server HTTP **nativo in-process** OpenAI/Anthropic-compatible (vedi sotto) |
| **Distribuito** | inferenza distribuita su più Mac (pipeline a range di layer, vedi sotto) |
| **Benchmark** | nativo: prefill + generazione (token/s) del motore in-process a contesti crescenti, con grafico (Swift Charts) |
| **Diagnostica** | dump dei token e del chat template (tokenizer nativo, niente sottoprocessi) |

### Quick start

```sh
make                  # swift build
make xcodeproj        # (re)generate DwarfStar.xcodeproj (xcodegen) — run after adding files
swift run DwarfStar   # launch the app
make test             # unit tests
```

In the app: choose the GGUF with **Sfoglia** (App Sandbox: typed paths won't
open — the picker grants a security-scoped bookmark that persists across
launches), press **Carica modello**, then chat. **Thinking** toggles
chain-of-thought; **Stop** cancels (the next turn rebuilds the KV if needed).

### HTTP server (native, in-process)

The Server tab starts an OpenAI/Anthropic-compatible server on
`Network.framework` — no subprocess, and the GGUF weights are mmap-shared with
the chat engine (no second copy in RAM). One request at a time (single model).

| Method | Path | API |
|---|---|---|
| GET | `/v1/models`, `/v1/models/{id}` | OpenAI |
| POST | `/v1/chat/completions` | OpenAI chat (stream + non) |
| POST | `/v1/responses` | OpenAI Responses (stream + non) |
| POST | `/v1/completions` | OpenAI legacy (stream + non) |
| POST | `/v1/messages` | Anthropic Messages (stream + non) |

```sh
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","stream":true,"messages":[{"role":"user","content":"Ciao"}]}'
```

### Distributed inference (multiple Macs)

Pipeline parallelism by contiguous layer ranges, modelled on `ds4_distributed.c`:
each **worker** owns a layer slice (weights *and* KV shard — allocated only for
its slice), the **coordinator** owns embedding, sampling and the prompt. The HC
hidden state (`nHC×nEmbd` floats, transported at 32/16/8 bit) flows through the
workers per token. Start the workers first, then the coordinator; the route must
cover all 61 layers contiguously (validated).

Per-node win: each worker streams only ~1/N of the experts, so its hot working
set is N× more likely to fit in RAM → fewer SSD page faults. Prefill runs in
chunks (default 32 tokens/frame); optional worker→worker forwarding halves the
hops (needs the coordinator's LAN address for the return path).

## Layout

```
Makefile / Package.swift / project.yml    build, SwiftPM package, xcodegen spec
Sources/
  DS4Core/        engine core: GGUF mmap parser, BPE tokenizer (control tokens),
                  sampler, model shape, chat/tools rendering + DSML parser
  DS4Metal/       Metal runtime + kernels + decode graph: StreamingDecoder
                  (forward/prefill/slice), expert slot-cache, usage stats,
                  GGUF weight loaders (no-copy mmap)
  DS4Engine/      InferenceService (actor, event stream), ToolRegistry, agents,
                  ProjectCache, Diagnostics, Distributed/ (protocol, transport,
                  worker, coordinator)
  DS4Demo/        CLI demo: Metal self-test + GGUF token streaming
  DwarfStar/      SwiftUI app (one folder per tab)
metal/            kernel sources (source of truth; embedded by make embed-kernels)
templates/        the model's chat template, re-written as commented Jinja
scripts/          GGUF analysis tools (spectrum, graph export) + kernel embedding
docs/             detailed documentation (IT)
```

## CLI demo

```sh
swift run DS4Demo                  # Metal bring-up + GPU self-test
swift run DS4Demo <model.gguf> 4   # stream 4 tokens through StreamingDecoder
```

## Packaging a .app

```sh
make app          # -> build/DwarfStar.app (release, ad-hoc signed)
open build/DwarfStar.app
```

For distribution, sign with a Developer ID and notarize (see
`packaging/make_app.sh`). The sandbox entitlements include
`network.client` + `network.server` (HTTP server / distributed) and
user-selected file access with app-scope bookmarks (models, projects).

## Status

**Working** (verified on a MacBook Pro M1 Pro 16 GB): model load + streaming
chat on the 2-bit GGUF, thinking, multi-turn KV reuse, DSML tool calling with
built-in tools (clock, calculator, add/subtract/multiply, project tools),
agents + per-agent expert profiles, project library, tuning (expert cache),
native HTTP server (OpenAI `chat/completions` verified end-to-end, Anthropic
`messages` verified streaming).

**Implemented, needs on-device validation**: `/v1/responses`, distributed
inference (protocol + worker/coordinator + UI are in place; numerical parity
and multi-Mac runs not yet verified).

**Known gaps**: decode on a 16 GB machine is I/O-bound (~57% expert gather) —
that is the physics of streaming 284B from SSD, and the distributed mode is the
intended mitigation. No subprocess-driven panels remain: server, distributed
and benchmark are all native in-process.
