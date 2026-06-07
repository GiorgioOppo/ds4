# DS4-gui

Native Swift / SwiftUI front-end for the **ds4 (DwarfStar)** DeepSeek V4
inference engine.

This folder is self-contained and does **not** modify the upstream engine. The
C / Objective-C / Metal engine in the parent directory is reused unchanged and
driven in-process through its narrow public boundary, [`ds4.h`](../ds4.h).

## Approach

```
DwarfStarApp (SwiftUI)        ← GUI (chat, models, server/agent, bench) — later phases
        │
   DS4Kit (Swift)             ← idiomatic bridge: engine, session, streaming
        │
   CDS4 (C module)            ← exposes ds4.h + a small shim to Swift
        │
   libDS4Engine.a             ← the upstream engine, compiled UNCHANGED
   (ds4.c, ds4_metal.m, ds4_ssd.c, ds4_distributed.c)
```

The engine is compiled with the **exact upstream Metal flags** (see the
`Makefile`), so in-process inference is byte-identical to the upstream `./ds4`
binary. Correctness is the project's #1 rule; the GUI must not perturb the
inference path.

### Key integration facts (validated against the engine source)

- **Metal kernels are compiled at runtime** from the per-kernel files under
  `metal/`, found relative to the working directory or via `DS4_METAL_*_SOURCE`
  env overrides (`ds4_metal.m`, function `ds4_gpu_full_source`). The shim
  `ds4gui_set_metal_source_dir()` points all 19 overrides at a chosen folder, so
  a bundled `.app` can ship `metal/` in its Resources.
- **One engine per process.** The Metal backend keeps global state (device,
  queue, ~120 pipelines, expert cache), so the model is loaded once. The GUI
  will not run the in-process engine and a `ds4-server` subprocess at the same
  time (that would load the model twice → OOM on large quants).

## Layout

```
DS4-gui/
  Makefile                       build libDS4Engine.a (+ drive swift build / xcodegen)
  Package.swift                  SwiftPM package (open in Xcode or build via CLI)
  project.yml                    xcodegen spec for the standalone .xcodeproj
  DwarfStar.xcodeproj            generated, clickable; builds the pure-Swift engine
                                 (DS4Core+DS4Metal+DS4Demo) with NO external links
  Sources/
    DS4Core/                     PURE-SWIFT engine core (GGUF, tokenizer, KV cache,
                                 sampler, model shape) — no C, no external links
    DS4Metal/                    PURE-SWIFT Metal runtime + kernels + graph:
                                 GPUTensor/GraphContext, decode layer, DSV4Decoder,
                                 StreamingDecoder, GGUF weight loader
    DS4Demo/                     pure-Swift CLI demo (Metal self-test + GGUF stream)
    CDS4/                        C module: ds4.h/ds4_ssd.h (symlinked) + shim
    DS4Kit/                      Swift bridge over the C engine (GUI in-process path)
    DwarfStar/                   SwiftUI chat app (ChatStore + views)
    ds4gui-smoke/                smoke test executable
  metal/                         vendored Metal kernel sources (copied in;
                                 the engine compiles them at runtime)
  enginelib/                     generated C engine static library (gitignored)
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

The kernel sources are embedded via `Sources/DS4Metal/KernelSources.swift`
(generated from `metal/*.metal` by `make embed-kernels`). `metal/` stays the
source of truth; rerun that after editing any kernel.

## Build & run (full SwiftUI GUI over the C engine)

The GUI app still drives the original C `./ds4` engine, so it needs the static lib:

```sh
cd DS4-gui
make            # builds enginelib/libDS4Engine.a, then `swift build`
make smoke      # build + run the smoke test
swift run DwarfStar   # launch the SwiftUI chat app
```

In the app: set the GGUF path, press **Carica modello**, then chat. The Metal
kernels are **embedded in the binary** (no kernel folder to set). Toggle
**Thinking** for chain-of-thought, **Stop** to cancel a generation. A complete
model is required to load (see below).

The smoke test validates the Swift → C bridge. With no model present it confirms
the build/link wiring. To run a real greedy generation, point it at a complete
ds4 GGUF:

```sh
swift run ds4gui-smoke ../ds4flash.gguf --prompt "Salutami in una frase."
```

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
copies the required `metal/` kernels into `Contents/Resources/metal` (the app
points the engine at them via `AppEnvironment` / `ds4gui_set_metal_source_dir`),
optionally bundles any built `ds4*` helper binaries under `Resources/bin`, and
ad-hoc code-signs the result so it runs locally.

For distribution, sign with a Developer ID identity and notarize:

```sh
DS4_SIGN_IDENTITY="Developer ID Application: NAME (TEAMID)" make app
ditto -c -k --keepParent build/DwarfStar.app DwarfStar.zip
xcrun notarytool submit DwarfStar.zip --apple-id … --team-id … --password … --wait
xcrun stapler staple build/DwarfStar.app
```

Drop an `AppIcon.icns` into `packaging/` to give the app an icon.

## Status

- **Phase 0 (scaffolding & build integration) — done.** Engine compiles into a
  static library; the Swift package builds and links against it; the Metal
  source directory is wired; the smoke executable runs.
- **Phase 1 (DS4Kit async bridge) — done.** `InferenceService` actor serializes
  engine access, streams reasoning/text as an `AsyncThrowingStream`, supports
  sampling, multi-turn history with KV reuse, and cooperative cancellation.
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
