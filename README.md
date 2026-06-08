# DS4-gui

Native Swift / SwiftUI front-end for **DeepSeek V4 (DwarfStar)**, with a
**pure-Swift inference engine** (a faithful port of the upstream `ds4.c` /
`ds4_metal.m`). No C engine, no prebuilt static lib, no external links.

> 📖 **Documentazione dettagliata (IT):** [`docs/DOCUMENTAZIONE.md`](docs/DOCUMENTAZIONE.md)
> (uso, demo e UI) · [`docs/ARCHITETTURA-MOTORE.md`](docs/ARCHITETTURA-MOTORE.md)
> (interni del motore: encoder, decoder, MoE, NSA, streaming).

## Approach

```
DwarfStarApp (SwiftUI)        ← GUI (chat, models, server/agent, bench)
        │
   DS4Engine (Swift)          ← InferenceService: prompt → event stream
        │
   DS4Core + DS4Metal (Swift) ← pure-Swift engine: GGUF, tokenizer, sampler,
                                 Metal runtime + kernels, decode graph
        │
   metal/*.metal              ← kernels, embedded in the binary at build time
```

The engine is a faithful Swift reimplementation; correctness is the project's
#1 rule, validated against the C originals in `Tests/DS4CoreTests/`.

### Key facts

- **Metal kernels are embedded in the binary** (`Sources/DS4Metal/Runtime/KernelSources.swift`,
  generated from `metal/*.metal` by `make embed-kernels`). They are compiled at
  runtime by `MetalRuntime`; no on-disk `metal/` folder is needed to run.
- **One engine per process.** The Metal backend keeps global state (device,
  queue, pipelines, expert cache), so the model is loaded once. The GUI will not
  run the in-process engine and a `ds4-server` subprocess at the same time (that
  would load the model twice → OOM on large quants).

## Layout

```
DS4-gui/
  Makefile                       drive swift build / test / xcodegen / packaging
  Package.swift                  SwiftPM package (open in Xcode or build via CLI)
  project.yml                    xcodegen spec for the standalone .xcodeproj
  DwarfStar.xcodeproj            generated, clickable; builds the whole pure-Swift
                                 stack with NO external links
  Sources/
    DS4Core/                     PURE-SWIFT engine core (GGUF, tokenizer, KV cache,
                                 sampler, model shape)
    DS4Metal/                    PURE-SWIFT Metal runtime + kernels + graph:
                                 GPUTensor/GraphContext, decode layer, DSV4Decoder,
                                 StreamingDecoder, GGUF weight loader
    DS4Engine/                   inference service backing the GUI (InferenceService,
                                 downloader, diagnostics)
    DS4Demo/                     pure-Swift CLI demo (Metal self-test + GGUF stream)
    DwarfStar/                   SwiftUI app (ChatStore + chat/server/bench/diag views)
  metal/                         Metal kernel sources (embedded into the binary
                                 via make embed-kernels; the runtime compiles them)
```

## Pure-Swift engine (.xcodeproj, no external links)

The DeepSeek-V4 engine has been reimplemented in pure Swift (`DS4Core` + `DS4Metal`,
dispatching the real `metal/` kernels). It has **no external links** — no C engine,
no prebuilt static lib — so it builds in a clean, clickable Xcode project:

```sh
cd DS4-gui
make xcodeproj         # (re)generate DwarfStar.xcodeproj via xcodegen
make xcode             # generate + open it in Xcode
# or build/run the demo from the CLI:
swift run DS4Demo                         # Metal bring-up + GPU self-test
swift run DS4Demo <model.gguf> 4          # + stream 4 tokens (StreamingDecoder)
```

`DS4Demo` brings up the Metal runtime (kernels are **embedded in the binary** —
no `metal/` folder needed at runtime), runs a GPU self-test, and — given a GGUF —
streams tokens through `StreamingDecoder` (per-layer load/compute/evict, so the
164GB model fits in 16GB).

The kernel sources are embedded via `Sources/DS4Metal/Runtime/KernelSources.swift`
(generated from `metal/*.metal` by `make embed-kernels`). `metal/` stays the
source of truth; rerun that after editing any kernel.

## Build & run (full SwiftUI GUI)

The GUI app is driven by the pure-Swift engine — no static lib, no C:

```sh
cd DS4-gui
make                  # swift build
make test             # run the unit tests
swift run DwarfStar   # launch the SwiftUI chat app
```

In the app: set the GGUF path, press **Carica modello**, then chat. The Metal
kernels are **embedded in the binary** (no kernel folder to set). Toggle
**Thinking** for chain-of-thought, **Stop** to cancel a generation. A complete
model is required to load (see below).

> The parent project currently has only a partial download
> (`gguf/*.gguf.part`). Finish a model download first, e.g.
> `cd .. && ./download_model.sh q2-imatrix`.

## Packaging a .app

```sh
cd DS4-gui
make app          # -> build/DwarfStar.app (release build + bundled metal/, ad-hoc signed)
open build/DwarfStar.app
```

`packaging/make_app.sh` builds the release executable, assembles the bundle,
copies the `metal/` kernel sources into `Contents/Resources/metal` (paths
resolved via `AppEnvironment`), optionally bundles any built `ds4*` helper
binaries under `Resources/bin`, and ad-hoc code-signs the result so it runs
locally.

For distribution, sign with a Developer ID identity and notarize:

```sh
DS4_SIGN_IDENTITY="Developer ID Application: NAME (TEAMID)" make app
ditto -c -k --keepParent build/DwarfStar.app DwarfStar.zip
xcrun notarytool submit DwarfStar.zip --apple-id … --team-id … --password … --wait
xcrun stapler staple build/DwarfStar.app
```

Drop an `AppIcon.icns` into `packaging/` to give the app an icon.

## Status

- **Phase 0 (scaffolding & build integration) — done.** Pure-Swift SwiftPM
  package + standalone `.xcodeproj`; the Metal kernels are embedded in the binary.
- **Phase 1 (async inference service) — done.** `InferenceService` actor (in
  `DS4Engine`, over the pure-Swift `DS4Core`/`DS4Metal` engine) serializes engine
  access, streams reasoning/text as an `AsyncThrowingStream`, supports sampling
  and cooperative cancellation.
- **Phase 2 (chat GUI) — initial app done.** `DwarfStar` SwiftUI app: model-load
  screen, streaming chat, collapsible reasoning, thinking toggle, stop/new chat.
- **Phase 3 (model management) — done.** GGUF discovery in `../` and `../gguf`,
  selectable list with sizes, SSD streaming toggle + cache budget (e.g. `32GB`),
  a download sheet that runs `download_model.sh` with live output, and a KV
  memory estimate shown after load.
- **Phase 4 (server/agent control) — done.** Sidebar shell. The Server panel
  launches `ds4-server` as a subprocess (host/port/ctx/CORS/disk-KV/streaming)
  with live log and cooperative stop, warns about the one-model-in-RAM
  constraint, and launches the interactive `ds4-agent` in Terminal.
- **Phase 5 (benchmark & diagnostics) — done.** Benchmark panel runs `ds4-bench`,
  parses its streamed CSV live, and charts prefill/generation throughput across
  context frontiers (Swift Charts). Diagnostics panel runs `ds4 --dump-tokens`.
  A shared `ProcessStream` helper backs every subprocess panel.
- **Phase 6 (packaging) — done.** `make app` builds a release `DwarfStar.app`
  with `metal/` bundled in Resources, bundle-aware paths (`AppEnvironment`), an
  Info.plist, and ad-hoc code signing; verified valid with `codesign`.

> All six phases build, link, and package cleanly. What has **not** been run in
> this environment: actual generation, the server subprocess, and the
> benchmark/diagnostics runners — only a partial GGUF is present, so verify those
> on a machine with a complete model. The server/bench/diagnostics panels also
> need the `ds4*` binaries built (`make` in the parent project).
