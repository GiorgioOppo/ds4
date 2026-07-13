# Tests

Unit tests for the pure-Swift engine layers. **Correctness is rule #1** in this
project: these tests validate the Swift port against the original C reference and
against CPU-faithful implementations where useful.

- **`DS4CoreTests/Core/`** covers deterministic formats, tokenization,
  conversation rendering, sampling, model shapes, and storage planning.
- **`DS4CoreTests/Metal/`** covers individual GPU kernels, graph composition,
  decode/cache behavior, model loading, and runtime creation.
- **`DS4CoreTests/Engine/`** covers distributed protocol, persistence, project
  safety, model management, diagnostics, and tools.

```sh
make test        # or: swift test
```

## Running From Xcode

The tests are also wired into the generated `.xcodeproj` as the `DS4CoreTests`
logic-test bundle, without an app host. After generating the project, open
`DwarfStar.xcodeproj` and press **Cmd+U**. The `DwarfStar` scheme has its Test
action connected to `DS4CoreTests`.

```sh
make xcodeproj
xcodebuild test -project DwarfStar.xcodeproj -scheme DwarfStar -destination 'platform=macOS'
```

## Metal prerequisites and skips

Metal tests may skip when the host exposes no Metal device. A number of legacy
tests also still use a developer-specific `metalDir` or production-GGUF path;
those skip when the fixture is absent and should be migrated to embedded kernels
or compact fixtures. A skip is not a pass and must be reported separately.

See [`METAL-TESTS.md`](METAL-TESTS.md) for the full skip, parity, and numerical
comparison conventions. Every test subdirectory has a local README describing
its scope and fixture rules.
