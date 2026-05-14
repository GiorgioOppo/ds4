# DeepSeek V4 on macOS

Swift + Metal port of the [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
Mixture-of-Experts transformer for Apple Silicon. Ships both a command-line
binary (`deepseek`) and a native SwiftUI macOS app, and can stream-load the
~142 GB **V4-Flash** weights on a 16 GB Mac thanks to a per-layer rotating
buffer.

> **Experimental.** V4-Pro itself (1.6T parameters, ~800 GB at FP4) does not
> fit in any Mac's unified memory; the realistic on-device target is
> **DeepSeek-V4-Flash** (284B / 13B activated). Same code, different config
> and weights.

🇮🇹 [Versione italiana](README.it.md) · 🏗 [Architecture deep-dive](docs/ARCHITECTURE.md)
· 🧪 [Testing](docs/TESTING.md) · 🛠 [Developing](docs/DEVELOPING.md)

---

## System requirements

| What | Minimum | Recommended |
|---|---|---|
| **CPU/GPU** | Apple Silicon (M1, M2, M3, M4…) | M-Ultra / M-Max |
| **macOS** | 14.0 Sonoma | 15.x |
| **RAM (unified)** | 16 GB (V4-Flash, streaming) | 64+ GB |
| **Disk** | 150 GB free for V4-Flash weights | NVMe SSD |
| **Tooling** | Swift 5.10 / Xcode 15+ | Xcode 16, Homebrew, Python 3 |

The loader picks a strategy automatically based on available RAM:

| Available RAM | Strategy | Behaviour |
|---|---|---|
| ≥ 192 GB | `preload` | Whole model resident, fastest |
| 32–192 GB | `mmap` | OS pages in on demand, fast after warm-up |
| 16–32 GB | `streaming` | One layer's shard at a time, slower first token |

Intel Macs are not supported (the Metal pipelines require `bfloat`-capable
hardware and `Sources/DeepSeekKit/Device.swift` will refuse to initialise).

---

## 1. Download the project

```bash
git clone https://github.com/giorgiooppo/DeepSeek-V4-Pro-MacOS.git
cd DeepSeek-V4-Pro-MacOS
swift package resolve
```

The repository does **not** ship the model weights or the tokenizer — those
files are gitignored. The next step downloads them separately.

## 2. Download the weights

The recommended checkpoint is **DeepSeek-V4-Flash** in its native HuggingFace
layout (FP8 attention + FP4 experts). The Swift loader reads that layout
directly — there is no conversion step.

```bash
pip install --upgrade huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
    --local-dir ~/Downloads/V4-Flash-HF
```

After the download, `~/Downloads/V4-Flash-HF/` must contain:

- `config.json`, `generation_config.json`
- `tokenizer.json`, `tokenizer_config.json`
- `model.safetensors.index.json`
- 46 `model-NNNNN-of-NNNNN.safetensors` shards (~142 GB total)

The companion `converter` binary is **only** needed if you want to
transcode the checkpoint to BF16/INT8/INT4 for smaller-disk variants. See
[`docs/USAGE.md`](docs/USAGE.md) for that path.

## 3. Build

### CLI only (fastest)

```bash
swift build -c release
```

Produces:

- `.build/release/deepseek` — inference CLI
- `.build/release/converter` — offline weight transcoder

### GUI app (Xcode)

```bash
brew install xcodegen        # one-time
./Tools/generate-xcodeproj.sh
open DeepSeekV4Pro.xcodeproj
```

Pick the **`DeepSeekApp`** scheme (not "DeepSeekUI", which is the SPM
executable target — both names show up because they share sources) and
press ⌘R.

---

## 4. Quick start

### CLI

```bash
./.build/release/deepseek ~/Downloads/V4-Flash-HF \
    "What is the capital of France?" \
    --mode chat --max-tokens 50 --temperature 0.7
```

Tokens stream to stdout as they're sampled. The first token can take
30 s – 3 min on a 16 GB Mac while the streaming loader warms up the layer
cache; subsequent tokens are faster.

### GUI

1. Launch the app from Xcode (⌘R) or by opening the built `.app`.
2. Pick the model folder (`~/Downloads/V4-Flash-HF`) when the picker
   appears. Recent folders are remembered.
3. Wait for the prefill indicator to finish — it shows the elapsed
   seconds live.
4. Type your message in the composer and press Send (or ⌘↩).

---

## 5. CLI reference

```
deepseek <model-dir> "<prompt>" [options]
```

Two positionals, the second optional only in diagnostic modes.

### Generation flags

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--mode` | `raw` \| `chat` | `chat` | `raw` prepends only BOS; `chat` applies the V4 chat template. |
| `--thinking` | `off` \| `high` \| `max` | `off` | Chat-mode reasoning budget. `off` appends `</think>` and the model answers directly; `high` appends `<think>` so the model emits a reasoning block first; `max` also prepends the REASONING_EFFORT_MAX system block. |
| `--temperature` | float | `1.0` | Sampling temperature. **Set to `0.7`** — see "Recommended values" below. |
| `--max-tokens` | int | `32` | Maximum tokens to generate. |

### Loader / memory flags

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--load-strategy` | `auto` \| `preload` \| `mmap` \| `streaming` | `auto` | Force a specific loader path. |
| `--force-load` | flag | off | Bypass the conservative RAM safety checks (shard > 70 % of RAM, total > 25× RAM). Use only if you know your system can tolerate aggressive paging. |
| `--max-seq-len` | int | from `config.json` | Override the KV-cache row count per layer. Lower = less RAM, shorter context. |
| `--max-batch-size` | int | from `config.json` | Override the batch dimension of the KV cache. V4-Flash ships with 1. |

### Diagnostic modes

| Flag | What it does |
|---|---|
| `--print-config` | Loads `config.json`, prints the resolved `ModelConfig` to stderr, exits. Verifies every key actually round-tripped instead of silently falling back to a default. |
| `--trace-norms` | Prints L2 norm + min/max/mean + NaN/Inf counters of the residual stream at key points in the forward pass. Useful for finding the layer where activations diverge. |
| `--list-tensors [PREFIX]` | Lists every tensor name in the checkpoint, optionally filtered by prefix. Pass `""` as prompt. |
| `--dump-tensor NAME[:row=R][:cols=A..B]` | Dequantizes one row slice of the named tensor and prints the float values, one per line. Defaults to `row=0`, `cols=0..32`. |

### Recommended values

- **`--temperature 0.7`**. V4-Flash's MoE routing under greedy argmax
  (temperature = 0) falls into self-reinforcing fixed points where the LM
  head loops on a single filler token (`好的好的好的…`, `_type_type_type…`).
  Values around 0.6–0.9 give the most coherent samples. The GUI clamps the
  slider to `[0.5, 1.0]` for the same reason.
- **`--mode chat --thinking off`** for short Q&A; `--thinking high` for
  problems where the model should "think out loud".

### Example: full inference

```bash
./.build/release/deepseek ~/Downloads/V4-Flash-HF \
    "Explain nuclear fusion in two sentences." \
    --mode chat \
    --thinking off \
    --temperature 0.7 \
    --max-tokens 256 \
    --max-seq-len 4096 \
    --max-batch-size 1
```

### Example: diagnostic

```bash
# Inspect what the loader actually parsed from config.json
./.build/release/deepseek ~/Downloads/V4-Flash-HF "" --print-config

# Find a tensor name
./.build/release/deepseek ~/Downloads/V4-Flash-HF "" \
    --list-tensors layers.0.

# Dequantize the first row of one expert's w1
./.build/release/deepseek ~/Downloads/V4-Flash-HF "" \
    --dump-tensor layers.6.ffn.experts.56.w1.weight:row=0:cols=0..64
```

---

## 6. The macOS app

### Model picker / loader

On first launch you choose the model directory through a standard
`NSOpenPanel`. Recently used folders are remembered (Preferences → Loading).
Throughout the load you see a `LoadPlan` summary (shard count, projected
RAM, chosen strategy) and a progress spinner. If anything goes wrong, the
panel offers **Try again**, **Force load**, or **Choose another folder**.

### Chat surface

- **Sidebar** lists every conversation, with a date stamp and a small
  spinner next to the one that's currently generating. Right-click →
  Delete. Cmd+N creates a new chat.
- **Prefill indicator**: a live counter (`Prefilling 256 tokens · 12.3s`)
  shows the model is making progress while the first forward pass runs.
- **Throughput bar** under the messages shows two monospaced lines once
  decoding starts:
  ```
  Prefill: 256 tok in 8.32s · 1850 tok/min
  Generation: 42 tok in 9.15s · 275 tok/min
  ```
  The generation line updates every ~0.5 s.
- **Token streaming**: tokens appear in the assistant bubble as they're
  sampled, exactly like in the CLI.
- **Reasoning blocks**: `<think>…</think>` content is rendered as a
  collapsible disclosure (a brain icon you can click to expand).
- **Send/Stop**: the Send button becomes Stop while a generation is
  in flight, mirroring the in-flight gating in `ChatStore`.

### Preferences

Four tabs. Changes take effect on the next Send (or, for `Model Config`,
the next model load).

| Tab | Controls |
|---|---|
| **Generation** | Temperature (slider 0.5–1.0, default 0.7), top-K (0 = disabled), top-P, max-tokens, thinking mode. |
| **Model Config** | Every field of `ModelConfig`. Writes to `~/Library/Application Support/<app>/config-overrides.json`; the loader honours `max_seq_len` and `max_batch_size` from it on the next load. |
| **Loading** | Loader strategy override, force-load toggle, last-loaded folder, recent folders, converter binary path. |
| **Storage** | Conversation history location, size on disk, "Reveal in Finder", "Clear all". |

### Convert sheet (offline quantization)

The toolbar's **Convert model…** action opens a sheet that drives the same
`converter` binary used from the CLI. Pick a source folder (HF native),
a destination, a target dtype (BF16 / F16 / INT8 / INT4 / INT2 / keep) and
a shard size. Progress and a live log stream into the sheet while it runs.

---

## 7. Troubleshooting

**The first token takes minutes.**
Expected under the `streaming` strategy on a 16 GB Mac. The loader has to
read each layer's ~3 GB shard from disk before processing it. Subsequent
tokens are much faster — the rotating slot stays warm.

**Build error: `precompiled file '…/ModuleCache/…' was compiled with module
cache path '…'`.**
The cached intermediate paths got stale, typically because the project
folder was moved (rename, iCloud sync, Trash). Wipe and rebuild:

```bash
rm -rf .build
swift package clean
swift build
```

For Xcode, also clear `~/Library/Developer/Xcode/DerivedData/DeepSeekV4Pro-*`.

**The model just loops a single token.**
You're sampling with `--temperature 0` or `0`. V4-Flash needs stochastic
sampling — pass `--temperature 0.7`. The GUI prevents this by clamping
the slider.

**"No Metal device" / Intel Mac.**
Apple Silicon is required. There's no fallback path.

**Out of memory at load.**
Try `--load-strategy streaming` to force the per-layer rotating loader, or
`--max-seq-len 2048 --max-batch-size 1` to shrink the KV cache.

---

## 8. License and credits

The Swift code in this repository is MIT-licensed (see [`LICENSE`](LICENSE))
and mirrors the upstream model license. The model weights and the Python
reference implementation in `Reference/inference/` belong to DeepSeek; see
their [Hugging Face card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
for license terms.

If you want to understand how the port works under the hood — kernel
mapping, residual amplification, streaming pool design, MoE dispatch —
read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For the contributor
workflow see [`docs/DEVELOPING.md`](docs/DEVELOPING.md), and for ready-made
prompts/recipes see [`docs/EXAMPLES.md`](docs/EXAMPLES.md).
