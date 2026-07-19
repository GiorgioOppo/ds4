**English** | [Italiano](METAL-TESTS.it.md)

# Running Metal Tests

Metal tests compare GPU results with compact CPU-faithful references. They are
part of `DS4CoreTests`, but require capabilities that may be absent in CI,
sandboxed processes, or non-Apple environments.

## Skip policy

- Use `XCTSkipUnless` or throw `XCTSkip` with a precise reason when no Metal
  device is available or an explicitly optional external fixture is absent.
- Do not catch pipeline compilation or numerical assertion failures and turn
  them into skips on a Metal-capable host.
- Do not silently `return` from a test: XCTest must report why it did not run.
- A skipped test provides no correctness signal. Release notes and summaries
  must list passes, failures, and skips separately.

The preferred runtime is `MetalRuntime()` with embedded kernels. Hard-coded
absolute paths to a developer checkout are legacy behavior and should not be
introduced in new tests.

## Numerical comparisons

1. Build deterministic, small input buffers.
2. Compute an independent CPU reference using the intended accumulation order.
3. Run the GPU kernel or graph operation.
4. Verify output shape and finiteness before value comparison.
5. Use exact equality for contracts documented as bit-identical; otherwise
   declare an absolute/relative tolerance appropriate to the data type.

Scheduling knobs such as simdgroup count should be tested at more than one
value when they promise identical output. Fused paths should be compared with
their unfused or CPU reference before performance measurements are trusted.

## Commands

```sh
swift test
```

From the generated Xcode project:

```sh
xcodebuild test \
  -project DwarfStar.xcodeproj \
  -scheme DwarfStar \
  -destination 'platform=macOS'
```

Use XCTest filtering for a focused area, for example
`swift test --filter MetalRoPETests`. Run the full suite after changing shared
runtime, buffer, graph, or kernel code.

