# Development guide

This is the practical checklist for changing the repository without breaking
module boundaries or generation flows. The full map is in
[STRUTTURA-PROGETTO.md](STRUTTURA-PROGETTO.md).

## Before making changes

1. Identify the target that owns the behavior.
2. Read the folder's `README.md` and the target's.
3. Check whether the file is generated.
4. Locate the tests in the mirrored domain.
5. Separate correctness, quality and performance in the proposal.

## Target boundaries

```text
DS4Core <- DS4Metal <- DS4Engine <- DwarfStar
    \----------- DS4Demo --------/
```

- Core does not import Metal, networking or SwiftUI.
- Metal knows nothing about Engine, HTTP or the GUI.
- Engine orchestrates the model but does not import SwiftUI.
- DwarfStar presents state and actions, without implementing GPU math.
- DS4Demo uses Core and Metal directly for diagnostics and performance.

## Where to place code

| Kind of change | Location |
|---|---|
| portable DTO or file format | `DS4Core` |
| tensor, GPU cache or dispatch | `DS4Metal` |
| inference API, persistence or networking | `DS4Engine` |
| SwiftUI state and views | `DwarfStar/Features/<Feature>` |
| CLI and audit | `DS4Demo` |
| GPU source | `metal` |
| tests | mirrored domain under `Tests/DS4CoreTests` |

Prefer one main type or one cohesive extension per file. Extensions
follow `Type+Responsibility.swift`.

## README policy

Every significant folder in the repository has a `README.md`. The local file
must answer four questions:

1. what the folder owns;
2. what the main files or types are;
3. what it may depend on;
4. how it is extended and how it is verified.

The local README must not duplicate huge tables or details bound to
change often. For a notable subsystem link a document under
`docs/`. After additions or moves verify that every new directory has
its own README.

## Generated files

Do not edit manually:

- `Sources/DS4Metal/Runtime/Generated/KernelSources.swift`;
- `DwarfStar.xcodeproj/project.pbxproj` as the primary source.

For kernels edit `metal/*.metal` and run `make embed-kernels`. For
the Xcode project edit `project.yml` and run `xcodegen generate`.

## Build workflow

```sh
swift build --disable-sandbox
swift test --disable-sandbox
swift build -c release --product DS4Demo --disable-sandbox
xcodegen generate
```

On macOS set `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`
if `xcrun` does not use the full Xcode installation.

## Kernel changes

Follow [BACKEND-METAL.md](BACKEND-METAL.md): `.metal` kernel, Swift wrapper,
graph composition, CPU/GPU tests, embedding and Release build form a single
unit of change.

## Pipeline changes

Keep separate:

- data and rendering;
- application state;
- KV/GPU state;
- weight I/O;
- sampling and presentation.

See [PIPELINE-INFERENZA.md](PIPELINE-INFERENZA.md). A new mode
must not create a second engine inside the GUI.

## Distributed protocol changes

1. Add the wire type under `Distributed/Protocol`.
2. Define limits and bound-checked decoders.
3. Add round-trip tests and malformed payloads.
4. Update coordinator and worker separately.
5. Bump `Dist.protocolVersion` for incompatible changes.
6. Update [INFERENZA-DISTRIBUITA.md](INFERENZA-DISTRIBUITA.md).

Never allow the wire to set arbitrary environment.

## GUI or server changes

Keep parsing/protocol, services, controllers and views in separate files.
Long-running actions must respect cancellation and actor isolation. See
[GUI-SERVER-E-API.md](GUI-SERVER-E-API.md).

## Documentation

When a behavior changes:

- update the owning folder's README;
- update the thematic document;
- fix examples and configuration in the main README;
- verify all relative links;
- avoid presenting future designs as features that already work.

Experimental documents must clearly state status, measurement date
and benchmark conditions.

## Final checklist

- [ ] Dependencies pointing in the right direction.
- [ ] No monolithic file grown with unrelated responsibilities.
- [ ] README present in new folders.
- [ ] SwiftPM manifest and XcodeGen up to date.
- [ ] Tests proportional to the risk executed.
- [ ] Embedded kernels regenerated when needed.
- [ ] No stale documentation paths.
- [ ] `git diff --check` clean.

The full testing strategy is in
[TESTING-E-VALIDAZIONE.md](TESTING-E-VALIDAZIONE.md).
