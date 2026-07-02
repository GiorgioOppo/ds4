# Tests

Unit tests for the pure-Swift engine layers. **Correctness is rule #1** in this
project: these tests validate the Swift port against the original C reference and
against CPU-faithful implementations where useful.

- **`DS4CoreTests/`** covers MoE matvec kernels, flash attention, normalization,
  RoPE, the decode graph, GGUF loading, tokenization, sampling, KV serialization,
  and the downloader.

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

> Metal kernel tests currently self-skip with `XCTSkipUnless` until their
> `metalDir`, currently a fixed absolute path at the top of each test file, points
> at the real `metal/` directory. They should be migrated to the embedded kernels
> through `MetalRuntime()` before being enabled in CI/Xcode.
