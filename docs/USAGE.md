# Usage

End-to-end runbook from an empty Mac to generated tokens.

## Prerequisites

- **macOS 14+ (Sonoma/Sequoia)** with Apple Silicon (M1/M2/M3/M4 series).
- **Xcode 15+** command-line tools (`xcode-select --install`).
- **git-lfs** + **huggingface-cli** for downloading checkpoints:
  ```bash
  brew install git-lfs huggingface-cli
  git lfs install
  ```
- **Disk space**:
  | Model | Input (HF) | Output (`bf16`) | Output (`keep`) |
  |---|---|---|---|
  | V4-Flash | ~140 GB | ~600 GB | ~140 GB |
  | V4-Pro   | ~900 GB | ~3.6 TB  | ~900 GB |
- **RAM**: ≥ 192 GB ideal (V4-Flash bf16 fits comfortably). 128 GB
  works for V4-Flash via mmap paging but with high SSD I/O. V4-Pro
  realistically needs Mac Studio M3 Ultra 512 GB and even then is slow.

## 1. Clone & build

```bash
git clone https://github.com/giorgiooppo/deepseek-v4-pro-macos.git
cd deepseek-v4-pro-macos
git checkout claude/convert-to-swift-FJDJC

# Plugin script must be executable.
chmod +x Plugins/MetalLibPlugin/build_metallib.sh

# First build is slow (~5-10 min): compiles 23 .metal files into
# default.metallib via xcrun metal + metallib.
swift build -c release 2>&1 | tail -20
```

Verify the metallib was produced:

```bash
find .build -name "default.metallib"
# should print at least one path under .build/release/...
```

Optional: run kernel tests.

```bash
swift test -c release 2>&1 | tail -20
```

## 2. Download weights

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /Volumes/DATA/checkpoints/V4-Flash-HF
```

Read `n_routed_experts` from the config (you'll pass it to the converter):

```bash
grep -E "n_routed_experts|num_experts" \
  /Volumes/DATA/checkpoints/V4-Flash-HF/config.json
```

## 3. Convert

Default: BF16 sharded, layer-aligned, 5 GB shards.

```bash
.build/release/converter \
  --hf-ckpt-path /Volumes/DATA/checkpoints/V4-Flash-HF \
  --save-path   /Volumes/DATA/checkpoints/V4-Flash-bf16 \
  --n-experts 256
```

### Flags

| Flag | Default | What it does |
|---|---|---|
| `--hf-ckpt-path <dir>` | required | HuggingFace V4 release directory |
| `--save-path <dir>` | required | Output directory |
| `--n-experts <N>` | required | Number of routed experts (from `config.json`) |
| `--model-parallel <K>` | 1 | Must be 1 (multi-rank sharding not supported) |
| `--target-dtype bf16\|f16\|keep` | `bf16` | See dtype trade-offs below |
| `--shard-size-gb <N>` | 5 | Max bytes per output shard |

### Target dtype trade-offs

| | `bf16` | `f16` | `keep` |
|---|---|---|---|
| Disk size | 4× input (~600 GB for V4-Flash) | same as bf16 | ≈ input (~140 GB) |
| Inference speed | fastest (native simdgroup matrix) | ~same | slower (shader dequant per element) |
| Memory pressure | highest | same | lowest |
| Precision | bf16 rounding | f16 rounding | bit-exact |

Recommendation: start with `keep` for first end-to-end run, switch to
`bf16` once you have disk + RAM headroom.

### Resume after interruption

The converter detects already-written shards on the output side and
skips them. Just re-run the same command:

```bash
.build/release/converter --hf-ckpt-path ... --save-path ... --n-experts 256
# Output starts with:
#   Resume: 14/133 shard(s) already on disk (66.0 GB skipped).
#   [15/133] model-00015-of-00133.safetensors — …
```

Resume detection requires **same input + same flags**. If you change
`--target-dtype` or `--shard-size-gb`, the new shard count won't
match the old filenames and the resume scan will start from 0. Wipe
the output dir manually in that case.

### Output structure

```
/Volumes/DATA/checkpoints/V4-Flash-bf16/
├── model-00001-of-00133.safetensors       # top-level (embed, head, norm, hc_head_*)
├── model-00002-of-00133.safetensors       # layer 0
├── …
├── model-00133-of-00133.safetensors       # MTP + final
├── model.safetensors.index.json           # name → shard map
├── tokenizer.json                         # copied from HF dir
└── tokenizer_config.json                  # copied from HF dir
```

## 4. Stage config.json for the runtime

The converter copies `tokenizer.json` / `tokenizer_config.json` but
**not** `config.json` (the model config the inference engine reads).
Copy it manually:

```bash
cp /Volumes/DATA/checkpoints/V4-Flash-HF/config.json \
   /Volumes/DATA/checkpoints/V4-Flash-bf16/config.json
```

## 5. Run

```bash
.build/release/deepseek \
  /Volumes/DATA/checkpoints/V4-Flash-bf16 \
  "Ciao, dimmi una poesia in tre versi" \
  --mode chat --max-tokens 100
```

### CLI flags

| Flag | Default | What it does |
|---|---|---|
| `<model-dir>` | required | Output dir of converter |
| `<prompt>` | required | Single prompt string |
| `--mode raw\|chat` | `chat` | `chat` adds BOS + role markers + EOS, parses `<think>` blocks. `raw` streams the model's literal output token by token |
| `--max-tokens N` | 32 | Token cap (stops earlier on EOS) |
| `--temperature T` | 1.0 | 0 = greedy argmax; otherwise Gumbel-max multinomial |

The first run is the slowest: the OS has to page in weights from SSD.
Subsequent runs reuse the file cache and are much faster.

For programmatic examples of the same operations (load a tensor,
dispatch a kernel, build a Layer), see [EXAMPLES.md](EXAMPLES.md).
For the conventions to follow when extending the project, see
[DEVELOPING.md](DEVELOPING.md).

## Troubleshooting

For an extended list of build-time errors and their causes, see
[DEVELOPING.md §8 Common pitfalls](DEVELOPING.md#8-common-pitfalls-we-hit-these--you-will-too).

| Symptom | Cause | Fix |
|---|---|---|
| `MTLLibraryErrorDomain code 6: no default library` | metallib not built | re-run `swift build -c release`; check `find .build -name '*.metallib'` |
| `Permission denied` on `build_metallib.sh` | not executable | `chmod +x Plugins/MetalLibPlugin/build_metallib.sh` |
| `no .safetensors files in …` | wrong `--hf-ckpt-path` | check the directory holds `*.safetensors` files (not just LFS pointers) |
| `all safetensors files … were LFS pointers` | git clone without LFS payload | re-download with `huggingface-cli download` |
| `No space left on device` mid-convert | output volume full | move `--save-path` to a bigger volume, or use `--target-dtype keep` for ~140 GB output |
| `N tensor name(s) were not found in the checkpoint` | name mismatch HF↔Swift | report the first few missing names; fixable in `Sources/DeepSeekKit/Assembly.swift` |
| `precondition failed: MLA decode expects seqlen == 1` | CLI loop bug | report output |
| Output garbage / repeats same token | sampling with `--temperature 0` and unusual prompt | try `--temperature 0.7` |
| First-token latency huge | cold SSD page cache | normal on first run; subsequent runs warm up |

When something fails, the most useful output is the last ~20 lines
before the crash plus the command you ran.
