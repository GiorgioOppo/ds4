# Machine Auto-Tuning Core

This directory contains the pure, architecture-neutral decision layer used by
the GUI machine auto-tuner. It deliberately does not load a GGUF, create a
Metal device, mutate `UserDefaults`, or read process environment variables.
The DwarfStar adapter is responsible for those effects and feeds independent
measurements into this layer.

`MachineAutoTune.swift` provides:

- a typed configuration and a RAM-aware manifest of exact-quality controls;
- compact, fail-closed quality signatures for bounded full-vocabulary logit
  traces: one frame per greedy token, a stable nonzero vocabulary size, and
  top-1/token agreement are mandatory;
- single-shot high-water, screening and order-balanced A/B/B/A gates;
- performance, secondary-metric, stability, free-memory and swap safeguards;
- an immutable low-RAM envelope derived from the already loaded root engine;
- an upward-first directional state machine for ordered ladders, plus neighbour
  selection and a mathematically correct median for coordinate search.

`MachineAutoTuneSwapCounter.swift` converts the host's cumulative swapout byte
counter into two explicit windows. Model init, warmup and the discarded warm
benchmark belong to a diagnostic load window; the promotion gate sees only the
subsequent measured steady-state benchmark. A fail-closed settling barrier after
the discarded primer validates and anchors the steady-state start sample; an
unavailable or invalid counter therefore aborts the observation instead of
weakening the guard. The end sample must be taken before quiesce/teardown.
Counter wrap is handled explicitly, while an unexplained counter regression is
treated as a reset and fails closed instead of being reported as zero swap.

The GUI uses `highWaterComparison`: the already loaded warm root and every
unique candidate are measured once, then stored in a run-local cache. The whole
eligible observation with the highest decode median is retained—metrics from
different observations are never combined. Promotion requires a strictly
higher decode result, at most 8% prefill regression, stability of at least 0.75,
the RAM floor, no more than 128 MiB of steady swapout, and bit-exact quality
against the immutable root. A repeated configuration is a cache hit and performs no reload.
This deliberately trades ABBA noise estimation for speed; an upward outlier can
make the record conservatively difficult to beat, but cannot weaken a safety or
quality gate.

Expert-cache slots use the directional state machine instead of a full sweep.
After `20→22` wins, it tests `24` without testing `18`; if `24` loses, that knob
stops and no larger manifest value is measured. If the first upward probe loses,
the immediate lower neighbour is the only fallback. Full sweeps remain reserved
for genuinely non-monotonic hardware grids.

`screening`, `repeatability` and `abba` remain available to the process-isolated
tuner and tests. The GUI no longer pretends that reused observations constitute
new ABBA pairs.

When the known-loadable root already leaves less than 10% free,
`MachineAutoTuneMemoryEnvelope` provides a constrained policy instead of
making tuning impossible. It requires at least 512 MiB while the known-loadable
root is still resident, anchors the whole run to the root's free-RAM level (with
one percentage point of VM-counter tolerance), and permits only candidates
whose known resident delta is zero or negative. Root restoration and these
bounded candidates retain the same 512 MiB known-loadable reserve after
teardown; unknown or memory-growing candidates require the stricter 12%/1.5 GiB
transient reserve. The normal speed, exact-quality, stability, and swapout gates
remain unchanged. The GUI must apply the same immutable envelope before init,
after init/warmup, and during final installation; it must never rebase the floor
after an intermediate promotion.

The standard safe manifest deliberately excludes `q8NSG`. Changing the Q8
reduction split can change floating-point accumulation order and therefore the
raw logit bits. The knob remains part of `MachineAutoTuneKnob` and
`MachineAutoTuneConfiguration` for explicit experimental manifests, but the
GUI's exact-quality default search does not explore it automatically.

Tests use synthetic observations and signatures, so gate and search behaviour
remain deterministic and run without model files or GPU access.

`MachineAutoTuneTransactionStore.swift` is the durable adoption boundary used
by the GUI after record-holder validation. The winner is installed and warmed
with the selected agent before preferences are committed, and the fully ready
chat engine must still satisfy the run's effective RAM floor and the 128 MiB
gate in its post-warmup steady-state probe. Its cold setup swap is reported
separately. A versioned atomic record lets the next app launch roll an
interrupted multi-key commit back to the complete initial snapshot. Its recovery
transitions are tested with isolated temporary defaults and record URLs.
