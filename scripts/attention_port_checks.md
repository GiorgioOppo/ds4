# Attention fusion checks without XCTest

`check_attention_port.swift` is a standalone executable using the public
`DS4Core` and `DS4Metal` APIs. It validates the fused output-B + HC update and the
paired Q4 prefill projections using synthetic weights. It does not load GGUF
models, start DwarfStar, change preferences, or download anything.

Build the library once, then link and run the checks:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
swift build -c release --target DS4Metal --scratch-path /private/tmp/ds4-attention-build
bash scripts/check_attention_port.sh /private/tmp/ds4-attention-build/release /private/tmp/ds4-attention-checks validate
```

The `DEVELOPER_DIR` override selects a full Xcode installation when
`xcode-select` still points at Command Line Tools. Adjust it to the installed
Xcode location; it is also needed to build the app's SwiftUI macros and run
XCTest with the current toolchain.

The first argument is the SwiftPM **products directory** containing `Modules`,
`DS4Core.build`, and `DS4Metal.build`, or the newer Swift Build backend's
`DS4Core.swiftmodule`/`DS4Metal.swiftmodule` directories and target-level `.o`
files. The `release` symlink works with either layout. `debug` also works for correctness checks;
use `release` for useful timings. The wrapper only links existing objects. Mode
`compile` stops before creating a Metal device; use it while other agents build
or edit kernels. Mode `benchmark` validates and then records paired timings.
GPU execution requires access to a real Metal device outside restricted sandboxes.
The compiled executable also accepts `--cpu-only` to check eligibility contracts
without creating a Metal device.

For a single suite after compilation:

```sh
env DS4_Q8_NSG=4 DS4_DENSE_Q4_NSG=4 /private/tmp/ds4-attention-checks/check_attention_port --suite decode --benchmark --pairs 8 --repeats 8
/private/tmp/ds4-attention-checks/check_attention_port --suite prefill --benchmark --pairs 8 --repeats 8
```

The wrapper runs decode first at the default NSG 4, then prefill, then validation
only at NSG 1, 2, and 8. Separate processes are required because the production
runtime caches these knobs. Prefill has no NSG parameter. Output-B
tests compare both the projection and four-stream HC result bit for bit with
the separate Q4_K/Q8_0 matvec + HC update. They cover 512→66 output tails and the
4096→4096 production shape, nested offset views, sentinel guards, immutable
inputs, repeated dispatches sharing scratch, two queued command buffers,
discarded uncommitted graphs, retained command-buffer resources, and rejected
disabled/invalid/overlapping calls. The reference Q4 tail weights include padding
for its existing inactive-SIMDgroup reads; output guards still bound the true row
count.

Prefill uses 4096→1024+512 at 32, 64, 128, and 256 tokens; 31 and 33 must return
false without modifying outputs or F16 scratch, then run the two reference
projections. Hardware outside the paired kernel's M1–M4 eligibility exercises
the fallback and prints an explicit skip for fused-only checks. A rejected
disabled/invalid/aliased call must be inert. Checks include guards, immutable
inputs, exact F16 activation staging, and repeated/queued scratch reuse.

Ordinary prefill output is compared with both existing GEMMs and an independent
sampled CPU oracle: scalar Q4 operand decoding matching the existing GEMM,
activation rounded once to F16, and a Double dot product. The scalar reference
preserves the high-nibble `half(d / 16)` boundary before the Float32 scale and
minimum arithmetic and final F16 weight rounding. The fixed elementwise bound is
`abs(error) <= 1e-4 + 1e-4 * abs(reference)`. Maximum error, RMSE, fraction of that
bound, and bit-difference counts are printed. A separate handcrafted subnormal
scale fixture requires bit parity with the old GEMM and uses the CPU oracle with
`1e-8 + 1e-4 * abs(reference)`. Its high-nibble F16 division underflow is preserved
intentionally, so the optimization does not change the existing model numerics.
Any failed bound terminates with nonzero exit status; the harness does not relax
tolerances after observing results.

Timings are **resident-kernel batch wall time**, including CPU encoding,
submission and wait, divided by the repeat count. They exclude weight creation,
quantization, allocation, readback validation, and output formatting. Three
warm-up pairs precede eight alternating AB/BA pairs. Logs retain each pair and
the median paired ratio. These are neither GPU timestamp measurements nor
end-to-end inference throughput: they cannot establish model tokens/second,
prefill latency, or quality. Run with other GPU workloads stopped and use the
same release objects and thermal conditions for comparisons.

Production defaults enable output-B + HC fusion for Q8 only. Q4 fusion is
validated but remains opt-in because it regressed in the M1 Max run below.
`DS4_FUSED_ATTN_OUT_HC=0` disables both; `=1` enables both. The setting is
refreshed at decoder creation. The harness explicitly enables each candidate
so correctness checks and measurements still cover Q4.
Paired Q4 prefill defaults on only for its supported M1–M4 devices and complete
production tiles. `DS4_PREFILL_Q4_PAIR=0` restores the two ordinary projections.

## Recorded run: 2026-09-21

Apple M1 Max, macOS 27.0 (26A428), Xcode 27.0 (27A266a), Apple Swift 6.4
(swiftlang-6.4.0.34.1), release objects. Eight alternating AB/BA pairs of eight
dispatch repetitions, after three warm-up pairs. Each number below is
milliseconds per operation, including CPU encoding/submission/wait.

| Operation | Reference median | Candidate median | Median paired speedup |
| --- | ---: | ---: | ---: |
| Output-B + HC Q8, 512→66 | 0.051286 | 0.043128 | 1.2022× |
| Output-B + HC Q8, 4096→4096 | 0.107620 | 0.092534 | 1.1342× |
| Output-B + HC Q4, 512→66 | 0.047940 | 0.051825 | 0.9612× |
| Output-B + HC Q4, 4096→4096 | 0.082227 | 0.088857 | 0.9381× |
| Paired Q4 prefill, 32 tokens | 0.504727 | 0.301747 | 1.6900× |
| Paired Q4 prefill, 64 tokens | 0.502482 | 0.350255 | 1.4638× |
| Paired Q4 prefill, 128 tokens | 0.573372 | 0.482253 | 1.1949× |
| Paired Q4 prefill, 256 tokens | 0.849906 | 0.812424 | 1.0464× |

All 16 decode fixtures (Q4/Q8 × two shapes × NSG 1/2/4/8) matched the
standalone projection and HC bits. All four paired prefill sizes also matched
the existing GEMMs bit for bit; the maximum sampled CPU-oracle error was
1.312e-6 within the declared bound. Both fallback sizes and the subnormal
fixture passed. Guards, immutable inputs, offset views, queued reuse, and
rejected-call checks passed. No numerical tolerances were changed after runs.

The paired speedup is the median of eight reference/candidate ratios, so it
need not equal the ratio of the two medians. These are one synthetic
microbenchmark run on one GPU and OS/compiler combination; small differences
can include scheduling noise. Q4 fusion was slower and is therefore opt-in.
No model was loaded, and these measurements establish neither tokens/second
nor end-to-end prompt latency. Full command output is retained in the ignored
workspace directory `build/attention-port-validation`.

The full release XCTest run on the final Q8-on/Q4-opt-in policy completed with
866 passes, 83 skips, and zero failures across 949 test cases (50 app tests and
899 Core/Metal/Engine tests). All seven attention-output/prefill XCTest cases
passed, including dispatch selection with the environment unset, `=0`, and
`=1`. The skips comprised 64 legacy kernel tests whose hard-coded external
source directory was absent, 17 missing or opt-in GGUF fixtures, and two manual
benchmarks. Those skips are not correctness evidence; the standalone checks
above use the shipped embedded kernels. No external model fixtures were added.

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
swift test -c release --scratch-path /private/tmp/swift-ds4-build \
  -Xswiftc -module-cache-path -Xswiftc /private/tmp/swift-ds4-module-cache \
  --no-parallel
```
