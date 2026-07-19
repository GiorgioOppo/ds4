**English** | [Italiano](TESTING-E-VALIDAZIONE.it.md)

# Testing and validation

The tests cover different levels: pure logic, formats, protocol, Metal
wrappers, graph composition and application services. A successful build is no
substitute for numerical tests; a GPU skip does not equal a passing test.

## Structure

```text
Tests/DS4CoreTests/
  Core/      tokenizer, conversation, formats, sampling, model, storage
  Metal/     runtime, kernels, graph, decode and weights
  Engine/    protocol, persistence, projects, download and tools
```

The historical target name is `DS4CoreTests`, but the target depends on Core,
Metal and Engine. The subfolders reflect the domain under test.

## Main commands

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build --disable-sandbox

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift test --disable-sandbox

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --product DS4Demo --disable-sandbox

xcodegen generate

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild build -project DwarfStar.xcodeproj -scheme DwarfStar \
  -destination 'platform=macOS' -derivedDataPath /tmp/DwarfStarDerivedData \
  CODE_SIGNING_ALLOWED=NO
```

Use the Release build for performance; the Debug build is for correctness and
diagnostics.

## Pure tests

The Core tests and most of the Engine tests do not require a GPU. They must
cover:

- valid and malformed input;
- size limits and overflow;
- serialization round-trips;
- persistent-format compatibility;
- explicit errors, not crashes or silent fallbacks;
- determinism when seed and input are fixed.

For protocols, always add tests for truncated payloads, impossible counts and
out-of-range fields.

## Metal tests

A typical kernel test prepares a small input, computes a CPU reference, runs
the wrapper and compares output within a tolerance. Cover at least:

- normal size;
- a tail not a multiple of the threadgroup when supported;
- offsets or views if the wrapper accepts them;
- every declared quantization;
- representative extreme values.

Tolerances must derive from the type and accumulation order, not be widened
until the test passes.

## GPU skips

Metal tests may be skipped when:

- no `MTLDevice` is available in the environment;
- the sandbox blocks access to the required compiler/runtime;
- an external fixture or a legacy kernel path is missing.

Every skip must include a readable reason. In CI or on a validation Mac,
distinguish between a suite that actually ran and a suite that was merely
discovered.

## Graph and decode parity

Graph tests compare full stages, not just primitives. For changes to route,
attention, compressor or MoE, verify:

1. the individual wrapper;
2. the graph composition;
3. the full layer;
4. multi-token forward when recurrent state is involved;
5. text/argmax on a real fixture, if available.

Prefill and decode must converge to the same state for an equivalent sequence,
except on explicitly approximated paths.

## Speed benchmarks

A useful benchmark keeps model, prompt, context, sampling, usage profile and
cache state constant. Record separately:

- load time;
- prefill tok/s;
- decode tok/s after warm-up;
- bytes read and effective bandwidth;
- memory pressure and swap;
- expert cache hit-rate.

Change a single knob per run. `DS4_PROFILE_ROUTE` alters timing and must not
be used for final throughput.

### A/B gate for Metal optimizations

For a new path selectable via `DS4_*`, use the process runner before promoting
it into the defaults:

```sh
scripts/metal_ab.sh model.gguf prompt.txt DS4_NUOVO_KNOB 0 1 8
```

The runner uses two separate processes with an identical prompt, greedy decode
and usage persistence disabled. It compares the generated ids and a limited
number of full logits vectors: the vectors are held copy-on-write during the
measurement, written only after the timers and analyzed via `mmap`. The report
distinguishes `PASS_EXACT`, `PASS_NUMERIC` and `FAIL` and includes prefill
tok/s, decode profile and regime. Set `DS4_AB_ATOL=0 DS4_AB_RTOL=0` for paths
that promise bit-for-bit parity; larger tolerances must be justified for
fusions with a different reduction order or explicitly lossy paths.

Page cache, temperature and memory pressure can favor one of the two
processes: repeat with `DS4_AB_ORDER=candidate-first` and do not promote
variations below the machine's noise floor. `python3
scripts/metal_ab_compare.py --self-test` validates the analyzer without
requiring Metal or a GGUF.

To search for a compound configuration across multiple knobs, use
`scripts/metal_autotune.py`: it runs one-dimensional sweeps/coordinate ascent,
confirms candidates in ABBA order, also compares each finalist against the
initial root, and rejects runs contaminated by RAM pressure or swap. The trace
format is validated fail-closed, and top-3, hash and finite-value counts are
recomputed from the `.f32` files, not taken on trust from the JSON. The full
procedure and commands are in [AUTOTUNING-METAL.md](AUTOTUNING-METAL.md).

## Next-token correctness benchmark

The correctness benchmark measures **top-1/top-2/top-3 accuracy under teacher
forcing** on a fixed text. For each position the model receives the real
prefix, ranks tokens by logit and checks whether the corpus's next token is
the first candidate, appears in the top two, or appears in the top three; in
the following step the real token is inserted regardless. The generation's
temperature and sampling do not participate in the metric. The benchmark seed
only chooses the corpus segments and their context lengths: given the same
input it makes the plan reproducible, but does not alter logits or the
candidate ranking. The three candidates are vocabulary tokens, not MoE
experts.

The multi-segment run requires the number of segments, minimum/maximum
context, maximum tokens per segment and a seed. Every first target must be
distinct; segments may nevertheless overlap. The context is drawn uniformly
within the effective interval and cannot exceed the target's index: the
per-segment maximum is therefore `min(massimoEffettivo, targetStart)`. The
context's starting point must remain non-negative. The planner prefers targets
that allow the requested maximum number of evaluations and falls back to the
short tail only when more segments are needed; each segment still evaluates
between one and the requested limit of tokens.

If the text produces `N` tokens and the minimum one-token prefix is used, the
verifiable predictions are `N - 1`: the first token forms the initial context
and is not a target. With an unevaluated prefix of `C` tokens, the available
targets become `N - C`, before any context or duration limits chosen by the
UI. These boundaries must remain covered by explicit tests, because evaluating
the prefix or skipping the last target introduces an off-by-one error in the
percentage. This is also the semantics of the historical single-segment
overload, which uses a fixed context.

Record at least:

- text/corpus or a stable content hash, and language;
- GGUF, tokenizer, build and full engine configuration;
- number of tokenized tokens and number of predictions actually evaluated;
- seed, requested/effective segments, context interval and per-segment
  maximum;
- correct tokens and overall accuracy for top-1, top-2 and top-3;
- starting point, context and evaluated tokens for each segment;
- any truncation imposed by the context or the chosen limit.

The three metrics must be nested in every observation, bucket and result:
`top-1 <= top-2 <= top-3`. Cumulative curves show the percentages from the
start of the run; local series measure only the current bucket. The last
bucket may contain fewer elements and must use its own real count as the
denominator for all three ranks. Even a low top-3 accuracy does not
automatically mean a semantically incorrect answer: natural texts often admit
multiple plausible continuations. The metric is mainly useful for comparing
builds, quantizations and optimizations on the same corpus.

Global accuracy must be computed as the sum of correct tokens divided by the
sum of evaluated tokens. Do not take a simple average of the segment
percentages: it would over-weight short samples. The per-segment chart serves
to show variability across the corpus, while the global KPIs stay weighted.

## Restructuring baseline

Verification performed on July 13, 2026:

- SwiftPM Debug build succeeded;
- Release build of `DS4Demo` succeeded;
- Xcode build of the app succeeded;
- 246 tests discovered/executed by the suite, 96 skips mostly tied to the
  Metal environment;
- two known assertions in `ProjectCacheTests.testSymlinksAreRefused`, already
  present in the baseline preceding the restructuring.

This snapshot is not a permanent waiver: when the environment changes or the
symlink test is fixed, update this document.

## Checklist for a change

- [ ] The code builds in SwiftPM Debug.
- [ ] The affected pure tests pass.
- [ ] The Metal tests were actually run on a Mac when necessary.
- [ ] The Release demo builds.
- [ ] The Xcode project was regenerated if files or folders changed.
- [ ] Lossless/lossy options are declared.
- [ ] The local README and the topic document are up to date.
- [ ] No generated file was edited by hand.

See also [GUIDA-SVILUPPO.md](GUIDA-SVILUPPO.md) and
[BACKEND-METAL.md](BACKEND-METAL.md).
