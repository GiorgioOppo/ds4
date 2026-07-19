**English** | [Italiano](README.it.md)

# DS4Demo Diagnostics

Focused helpers used by the command-line executable:

- `DemoLog.swift` keeps diagnostic logging on `stderr`, separate from generated
  text written to `stdout`. It also owns the opt-in bounded A/B logit trace:
  vectors are retained copy-on-write during inference and serialized only after
  the timed region for `scripts/metal_ab.sh`.
- `DiskBenchmark.swift` measures storage behavior relevant to streamed weights.
- `ModelDiagnostics.swift` reports GGUF tensor and tokenizer information.

Diagnostics may observe and report runtime behavior but should not silently
change generation settings. Keep output stable enough for benchmark comparison
and shell capture.

The expert-cache readout separates hits/misses on cacheable layers from bypass
selections and reports the resulting global hit-rate. Look-ahead bytes are
reported separately because they are real storage traffic hidden outside the
critical-path gather timer. Per-layer slot/RAM rows are snapshots of pools that
actually existed in the measured run, never a post-run allocation recomputed
from routing history that changed while the run was executing.
