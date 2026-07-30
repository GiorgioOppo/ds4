**English** | [Italiano](AUTOTUNING-METAL.it.md)

# Multi-parameter Metal autotuning

`scripts/metal_autotune.py` searches for a configuration with higher throughput
without promoting a candidate that degrades the numerical check. It is meant
for long measurements on a real Mac, not for CI.

## What it guarantees

The search starts from a complete configuration and changes one parameter at a
time. For each candidate it:

1. runs baseline and candidate in separate `DS4Demo` processes;
2. compares tokens, argmax, top-3 and full logits of the recorded frames;
3. immediately discards crashes, non-finite values, incomplete traces and
   regressions;
4. if the result is close or better, repeats in the reverse order;
5. computes the balanced geometric ratio of the AB and BA pairs;
6. promotes the value only beyond the minimum margin, 2% by default;
7. for ordered knobs, keeps going in the same direction until the first
   regression; for queue depth, cache slots and NSG it instead tries the whole
   small grid, because these parameters are not necessarily unimodal;
8. repeats all parameters over multiple passes to measure interactions.

The overall order for a finalist is therefore **ABBA**. The result is not a
Cartesian grid: it does not combine every value of every parameter with all the
others. The one-dimensional sweeps do, however, avoid stopping, for example, at
`PREAD_SPLIT=2` without measuring the value `4`. Coordinate ascent reaches a
local maximum with a tractable number of trials.

For parameters declared lossless the gate is `PASS_EXACT`: every Float32 bit
must match. `PASS_NUMERIC` is not enough, not even with zero tolerance. The
`--allow-numeric` mode is explicit and adds conservative requirements:
identical tokens, argmax and ordered top-3, zero non-finite values, all values
within `atol=1e-4, rtol=1e-5`, maximum absolute error `1e-3` and
aggregate/per-frame NRMSE no greater than `1e-5`.

The comparator does not trust the top-3 saved in the JSON: it recomputes
directly from the `.f32` files the top-3, finite values and FNV hash of every
frame. Overlapping offsets, stale metadata, truncated traces and inconsistent
token/argmax mappings are a fail-closed error. With the defaults `max-new=64`
and `trace-frames=64`, all the decisions that produced the measured tokens are
covered.

This is parity on the supplied prompts, not a universal semantic measure. That
is why the tuner excludes reducing the active experts, new quantizations,
`COMP_Q8` and the other lossy changes. Those must also use the app's
teacher-forced correctness benchmark.

## Profiles

| Profile | Parameters |
|---|---|
| `io` | `pread` split, cache slots, usage-driven allocation, expert look-ahead and dense-ahead |
| `standard` | `io` profile plus `MOE_NSG` and `DENSE_Q4_NSG`; this is the recommended starting point |
| `full` | adds the secondary lossless flags and the prefill knobs; can take many hours |
| `prefill` | union, chunk, route batch and FFN batch on a long prompt |
| `numeric` | only numerically-close knobs; requires `--allow-numeric` |

The standard profile does not change `RAW_RING=1`, the quantizations already
chosen or the number of experts. The default preset is the mixed-Q4/IQ2 one for
M1 Pro 16 GB, pread backend with `RAW_RING=1`; `DS4_*` values already exported
in the terminal override it. `--preset inherit` inserts no implicit values: to
avoid a false baseline it requires that every selected knob is already
explicitly present in the environment.

## Deterministic usage-imatrix

Cache allocation and look-ahead depend on routing history. With
`--usage-seed auto`, the tuner looks for the richest file for the same model
name in the DwarfStar Application Support folders, freezes a copy of it and
creates a private copy for each process. This way every A and B starts from the
same history and does not overwrite the GUI's.

`final-env.sh` also points to the frozen stable copy, so the configuration
truly reproduces the tests of the usage-dependent knobs.

If no seed is found, the usage-dependent parameters are skipped. It is possible
to pass an explicit file or to use `--allow-cold-usage`, but the latter mode
does not represent the GUI's behavior after multiple conversations.

## Recommended execution

Close the GUI and any applications using a lot of memory. Do not run the
benchmark while a download is active: a `.part` file modified in the last ten
minutes in the model directory automatically blocks the tuner. The check is
repeated before every process, so it also catches a download started after the
search has already begun.

Every run also records `memory_pressure` and the `vm_stat` counters: below 8%
free memory or beyond 128 MiB of new swapouts the candidate is rejected. The
thresholds are configurable, but it is not worth lowering them to choose
production defaults.

Prepare the prompt:

```sh
printf '%s\n' \
  'raccontami la storia di roma come se dovessi scrivere un libro di storia' \
  > /tmp/ds4-roma-prompt.txt
```

Start the standard profile:

```sh
cd "/Users/oppog/Documents/Project/DeepSeek v4 Metal/DeepSeek-V4-Pro-MacOS"

python3 scripts/metal_autotune.py \
  "/Users/oppog/Library/Containers/com.dwarfstar.app/Data/Library/Application Support/DwarfStar/models/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf" \
  /tmp/ds4-roma-prompt.txt \
  --profile standard \
  --context 100000 \
  --output /tmp/ds4-autotune-standard
```

`--context` must match the window actually used: KV cache and RAM budget change
the slot optimum. For a 100k configuration, do not promote a result measured
only with the 4096 preset.

Every candidate requires at least two full loads; promising ones require four.
Before starting, the search space can be inspected without building or running
inference:

```sh
python3 scripts/metal_autotune.py MODEL.gguf prompt.txt \
  --profile full --dry-run
```

For prefill you need a text that produces at least 1024–2048 tokens. The short
Roma prompt does not cross `PREFILL_CHUNK` boundaries and cannot choose that
value. The full profile can use two workloads:

```sh
python3 scripts/metal_autotune.py MODEL.gguf prompt-decode.txt \
  --profile full \
  --context 100000 \
  --prefill-prompt corpus-prefill-lungo.txt \
  --output /tmp/ds4-autotune-full
```

This is the command to use to evaluate **all the lossless knobs** of the
manifest. The two numerically-close knobs stay excluded; add `--allow-numeric`
only in a second, separate search if the tolerance gate is acceptable.

Decode remains the objective for the decode parameters; prefill becomes the
objective for its own parameters. The other metric is a secondary guard and
cannot regress beyond the configured threshold.

## Interruption and resume

Every process and every decision is saved with an atomic write. After
`Ctrl-C`:

```sh
python3 scripts/metal_autotune.py MODEL.gguf prompt.txt \
  --profile standard \
  --output /tmp/ds4-autotune-standard \
  --resume
```

Resume is refused if the binary, model, prompt, usage seed, initial
environment, manifest, tuner/comparator scripts or acceptance policy have
changed. A run ID is checkpointed before starting the process: even a
`kill -9` can at most leave an orphan folder, never break the resume.

## Results

The chosen directory contains:

- `report.md`: initial/final configuration and all decisions;
- `results.csv` and `results.json`: machine-readable data;
- `final-env.sh`: complete environment to load with `source`;
- `state.json` and `events.jsonl`: crash-safe checkpoint and journal;
- `runs/*/run.log`: log of every process and its effective environment.

The intermediate Float32 traces are deleted after the comparison, unless
`--keep-traces` is given. `final-env.sh` shows `VALIDATED` only after the
final initial-vs-final validation has passed the gate.

Applying the winners to a subsequent demo:

```sh
source /tmp/ds4-autotune-standard/final-env.sh
.build/release/DS4Demo MODEL.gguf 256 "@/tmp/ds4-roma-prompt.txt"
```

## Main operational options

| Option | Effect |
|---|---|
| `--knobs A,B,C` | use only the listed knobs |
| `--min-gain 0.02` | minimum ABBA margin for a promotion |
| `--max-passes 2` | maximum number of coordinate passes |
| `--context 100000` | context window of the configuration being optimized |
| `--cooldown 2` | pause between processes for memory/thermal pressure |
| `--usage-seed auto|off|PATH` | frozen routing history |
| `--min-memory-free-percent 8` | rejects runs that finish below the RAM threshold |
| `--max-swapout-mib 128` | rejects runs that cause too much swapout |
| `--allow-numeric` | enables the numerically-close manifest |
| `--keep-traces` | keeps the JSON and Float32 of every comparison |
| `--skip-build` | reuses `.build/release/DS4Demo` |
| `--self-test` | synthetic tests without model or GPU |

`--no-final-validation` is available only for interrupted explorations: the
result will have state `complete_unvalidated` and `final-env.sh` will stay
marked `NOT FINAL`.

Do not use `--allow-active-download` for a measurement meant for promotion: it
exists only for deliberate diagnostics, and the resulting report would be
contaminated by the external I/O.

## GLM 5.2 auto-tune

GLM has its own pragmatic auto-tune (GUI button "Auto-tune GLM" in the GLM
settings, or `DS4_GLM_AUTOTUNE=1` in the demo): it climbs one knob at a time
over the EXACT load-time knobs only (`DS4_MTLIO`, `DS4_STREAM_SLOTS`,
`DS4_GLM_EXPERT_ARENA`, `DS4_RESIDENT_LAYERS`, `DS4_GLM_FUSE`), reloading
the engine per configuration — the ~3 s streaming load makes per-config
reloads affordable, which the DeepSeek record-holder cannot do. Quality
knobs (active experts, lossy Q4) are never touched, and winners are applied
to the persisted GUI settings before the model is reloaded. There is no
transactional journal or stability metric yet: measurements are single-shot
and the champion threshold is 3% over run noise.
