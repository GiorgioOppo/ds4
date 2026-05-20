# Usage

End-to-end operational reference. Three tracks depending on what
you want to do:

1. **Local model in the GUI** — recommended.
2. **Remote model through OpenRouter in the GUI** — no GPU
   required, paid per token.
3. **Local model on the CLI** — for headless / scripted runs.

The CLI does not talk to OpenRouter (yet); remote inference lives
only in the desktop app.

For a beginner-friendly walkthrough in Italian, start with
[`ISTRUZIONI.md`](ISTRUZIONI.md). For ready-made recipes, see
[`EXAMPLES.md`](EXAMPLES.md).

---

## Prerequisites

- **macOS 14+** with Apple Silicon (M1/M2/M3/M4 series).
- **Xcode 15+** command-line tools:
  ```bash
  xcode-select --install
  ```
- For local inference: **150 GB free disk** for V4-Flash weights and
  **≥ 16 GB RAM** for streaming mode (≥ 64 GB recommended for
  comfortable speeds).
- For weight download: `huggingface-cli`:
  ```bash
  brew install huggingface-cli
  ```
- For the GUI app build: **XcodeGen**:
  ```bash
  brew install xcodegen
  ```

---

## 1. Build

```bash
git clone https://github.com/giorgiooppo/DeepSeek-V4-Pro-MacOS.git
cd DeepSeek-V4-Pro-MacOS

# Plugin script must be executable (one-time).
chmod +x Plugins/MetalLibPlugin/build_metallib.sh

# CLI binaries.
swift build -c release 2>&1 | tail -5

# GUI app.
./Tools/generate-xcodeproj.sh
open DeepSeekV4Pro.xcworkspace
# In Xcode pick the DeepSeekApp scheme, ⌘R.
```

Verify the Metal library compiled:

```bash
find .build -name "default.metallib"
```

Optional kernel test sweep:

```bash
swift test -c release 2>&1 | tail -20
```

---

## 2. Local models

### 2.1. Download weights

**Recommended path: native HuggingFace layout**, no conversion
required. The Swift loader reads FP8 attention + FP4 expert
checkpoints directly.

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir ~/Downloads/V4-Flash-HF
```

The destination must contain `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `model.safetensors.index.json`, and 46
`model-NNNNN-of-NNNNN.safetensors` shards.

### 2.2. (Optional) Convert to a compact variant

Only needed when you want smaller-disk variants (INT4 / INT8 etc.)
or BF16 for benchmarking — most users skip this step.

```bash
.build/release/converter \
  --hf-ckpt-path ~/Downloads/V4-Flash-HF \
  --save-path   ~/Downloads/V4-Flash-int4 \
  --n-experts 256 \
  --target-dtype int4 \
  --shard-size-gb 5
```

Then copy `config.json` next to the new shards (the converter does
not):

```bash
cp ~/Downloads/V4-Flash-HF/config.json \
   ~/Downloads/V4-Flash-int4/config.json
```

**Converter flags**

| Flag | Default | Notes |
|---|---|---|
| `--hf-ckpt-path <dir>` | required | HuggingFace V4 release directory. |
| `--save-path <dir>` | required | Output directory. Resumes existing shards on re-run. |
| `--n-experts <N>` | required | `n_routed_experts` from `config.json`. |
| `--target-dtype` | `bf16` | One of `bf16` \| `f16` \| `int8` \| `int4` \| `int2` \| `keep`. |
| `--shard-size-gb <N>` | 5 | Max bytes per output shard. |
| `--model-parallel <K>` | 1 | Multi-rank not supported. |

**Dtype trade-offs**

| dtype | Disk size for V4-Flash | Inference speed | Notes |
|---|---|---|---|
| `bf16` | ~600 GB | fastest (native simdgroup matrix) | benchmark baseline |
| `f16` | ~600 GB | ~same as bf16 | minor numerical difference |
| `keep` | ~140 GB | slower per-element dequant | bit-exact to HF release |
| `int8` | ~290 GB | between bf16 and keep | symmetric quant |
| `int4` | ~160 GB | similar to keep | 4-bit quant with scales |
| `int2` | ~95 GB | slowest dequant | experimental |

### 2.3. Load in the GUI

1. Launch the app. The chat surface appears immediately with a
   `No model loaded` banner above the composer.
2. Toolbar **Model** menu (leftmost) → **Choose model folder…** →
   pick `~/Downloads/V4-Flash-HF`.
3. The banner becomes `Loading <name>… <gb> GB across <n> shards
   · strategy: <preload|mmap|streaming>`.
4. When the banner disappears the model picker label shows the
   folder name and Send is enabled.

Recents are remembered automatically; next launch auto-loads the
last successful model (`AppSettings.lastModelDir`).

Use **Unload current model** from the same menu to free RAM
without quitting.

### 2.4. Non-V4 local models (preview)

The engine targets DeepSeek-V4 (the MLA + MoE + HC stack only
matches that architecture). Pieces that make *other* local
models a future possibility are now in place:

- **Chat template dispatcher**: drops the loaded model's
  Jinja2 chat template (Llama / Mistral / Qwen / ChatML) into
  the prompt path via `JinjaChatTemplate`.
- **Multi-format tokenizers**: BPE / SentencePiece / WordPiece
  resolved by `TokenizerLoader.load(tokenizerDir:)` from the
  files actually present in the folder.
- **GGUF reader (MVP)**: `GGUFFile` parses v2/v3 headers and
  returns zero-copy `Tensor` views for `F32 / F16 / BF16 / I32 /
  I8`. Quantized dtypes raise `GGUFError.unsupportedType`.

Inference for those models won't run today (no Llama transformer
kernels, no quant dequant kernels). See [`GGUF.md`](GGUF.md) for
what's missing.

### 2.5. Run on the CLI

```bash
.build/release/deepseek ~/Downloads/V4-Flash-HF \
  "What is the capital of France?" \
  --mode chat --thinking off \
  --temperature 0.7 --max-tokens 256
```

**Generation flags**

| Flag | Default | Notes |
|---|---|---|
| `--mode raw\|chat` | `chat` | `raw` only prepends BOS; `chat` applies the V4 template. |
| `--thinking off\|high\|max` | `off` | Reasoning budget for chat mode. |
| `--temperature T` | `1.0` | **Set to 0.7** — see § Sampling caveat. |
| `--max-tokens N` | `32` | Stops earlier on EOS. |

**Sampler flags (all default to "disabled")**

| Flag | Default | Notes |
|---|---|---|
| `--top-k K` | `0` | Keep only the top-K logits. |
| `--top-p P` | `1.0` | Nucleus mass. |
| `--min-p P` | `0.0` | Drop `p < P × max_p`. |
| `--tfs Z` | `1.0` | Tail-free sampling. |
| `--typical P` | `1.0` | Locally-typical mass. |
| `--repetition-penalty R` | `1.0` | HuggingFace-style on history ids. |
| `--frequency-penalty F` | `0.0` | OpenAI-style, scales with count. |
| `--presence-penalty P` | `0.0` | OpenAI-style, binary on presence. |
| `--mirostat TAU` | off | Mirostat v2 with target surprise τ. |
| `--mirostat-eta ETA` | `0.1` | Mirostat learning rate. |

The pipeline order matches `Sampler.sample`: temperature →
repetition → freq/presence → top-K → top-P → min-p → tfs →
typical → (Mirostat replaces the K/P/min-p/tfs/typical block
when enabled) → Gumbel-max multinomial.

**Loader flags**

| Flag | Default | Notes |
|---|---|---|
| `--load-strategy auto\|preload\|mmap\|streaming` | `auto` | Auto picks based on free RAM. |
| `--force-load` | off | Bypass RAM safety checks. |
| `--max-seq-len N` | from config | Override KV-cache row count. |
| `--max-batch-size N` | from config | Override batch dim. |

**Diagnostic flags**

| Flag | Behaviour |
|---|---|
| `--print-config` | Print resolved `ModelConfig`, exit. |
| `--trace-norms` | Print residual stream L2 / min / max / NaN counters per layer. |
| `--list-tensors [PREFIX]` | List every tensor name. Pass `""` as prompt. |
| `--dump-tensor NAME[:row=R][:cols=A..B]` | Dequantise one row slice and print floats. |

**Sampling caveat.** V4-Flash MoE routing under greedy argmax
(`--temperature 0`) falls into self-reinforcing loops where the LM
head repeats a filler token (`好的好的…`, `_type_type_type…`).
Use `--temperature 0.7` or in the GUI keep the slider in the
clamped `[0.5, 1.0]` window.

---

## 3. Remote models (OpenRouter)

OpenRouter is an OpenAI-compatible API gateway that routes to
Claude, GPT, DeepSeek, Llama, Mistral, and ~300 other models
through one URL and one key. The desktop app's remote backend
targets it specifically.

### 3.1. Add your API key

1. Sign up + get a key at <https://openrouter.ai/keys>.
2. App → **Settings (⌘,) → API Keys** tab.
3. Paste into the OpenRouter SecureField. **Save**.
4. (Optional) **Test** — pings `/auth/key` and shows a green
   "Key accepted" if it works.

The key is stored in the **macOS Keychain** under service
`com.deepseek.v4pro`, account `openrouter.apiKey`. It is never
written to a plist.

To rotate: open the same tab, **Delete**, paste the new one,
**Save**.

### 3.2. Pick a model

1. Toolbar **Model** menu → **Add OpenRouter model…**.
2. The sheet shows OpenRouter's full catalog. First open fetches
   it (~50 KB JSON); cached on disk under
   `Application Support/.../openrouter-catalog.json` for 24 h.
3. Use the search box to filter by provider (`anthropic`), name
   (`sonnet`), or slug fragment (`r1`).
4. Each row shows context length and per-token pricing
   (`$X.XX/M in · $Y.YY/M out`), or `free` for free-tier models.
5. Click a row → `ModelState.load` validates the key + flips the
   chat to that endpoint. Almost instant — no weights to map.

The chosen model appears under **Recent** and auto-loads at the
next launch (just like local folders).

### 3.3. Send and observe

The chat behaves identically to the local path: type, ⌘↩, watch
the bubble fill in. Differences specific to remote:

- Streaming uses SSE; the first token usually arrives in
  1–3 seconds depending on the upstream provider.
- **Reasoning content** (DeepSeek-R1, o-series) is captured in
  the `reasoning_content` field and rendered through the same
  brain-icon disclosure as local `<think>` blocks.
- **Cost** is reported by OpenRouter in `usage.total_cost`. The
  ThroughputBar shows `Turn cost: $0.0042` under the assistant
  bubble; a banner above the composer shows `Chat total: $0.013`
  cumulative across the conversation, persisted to disk.

### 3.4. Tool calling on remote

Same MCP plumbing as local. When the chat is on an OpenRouter
endpoint, the MCP catalogue (filtered through the agent's
allowlist if one is attached) is translated to the OpenAI `tools`
array. Tool calls fire HTTP round-trips up to 21 iterations per
turn; outputs splice back as `{role:"tool", tool_call_id, content}`
messages.

The synthetic `__delegate_to_agent` schema is **not** injected on
remote chats — cross-agent delegation runs only locally for now.

### 3.5. Reasoning mode on remote

The thinking picker above the composer maps as:

| Picker | Sent to OpenRouter |
|---|---|
| No think | (no `reasoning` field) |
| High | `reasoning: { effort: "medium" }` |
| Max | `reasoning: { effort: "high" }` |

Providers that don't honour the field silently ignore it.
DeepSeek-R1 and o-series respect it.

---

## 4. Native tools, Plan / Build, slash commands

The `DeepSeekTools` target ships a code-agent toolbox the model
can call directly (no MCP round-trip required). Full catalogue
in [`TOOLS.md`](TOOLS.md); the operating-mode gate that filters
mutating / dangerous tools is in [`AGENT-MODES.md`](AGENT-MODES.md).

### 4.1. Tool catalogue (today)

| Category | Tools |
|---|---|
| readOnly | `read`, `glob`, `grep`, `repo_overview`, `lsp` (stub) |
| planning | `plan`, `task`, `todo` |
| mutating | `write`, `edit`, `apply_patch` |
| dangerous | `shell`, `repo_clone` |
| network | `webfetch`, `websearch` |

Categories drive both the plan-mode filter and the permission
policy.

### 4.2. Plan vs Build

Every agent's `agentMode` is either `.build` (historical) or
`.plan` (read-only stance):

| Mode | readOnly | planning | mutating | dangerous | network |
|---|---|---|---|---|---|
| build | allowed | allowed | allowed (consent) | allowed (consent) | allowed (consent) |
| plan  | allowed | allowed | **denied** | **denied** | allowed (consent) |

Three entry points to switch:

1. **Settings → Agents → edit → Agent mode** segmented control
   (persists into `agents.json`).
2. **Toolbar Mode picker** in the chat — flips the active
   conversation's mode without editing the saved agent.
3. **`/mode plan`** / **`/mode build`** slash command in the
   composer.

### 4.3. Permission gate

Dispatching a mutating / dangerous / network tool walks this
ladder until it gets an answer:

```
mode filter
   ↓ (passes only if the mode allows the category)
PermissionStore durable default
   alwaysAllow → run
   alwaysDeny  → return ToolError.denied
   ask         → ↓
session cache (ToolRegistry)
   cached → run
   miss   → ↓
PermissionPromptView (modal)
   Deny | Allow once | Always allow
```

"Always allow" writes through to `PermissionStore` so the next
launch starts from the same answer. Reset all session grants
from **Settings → Permissions → Reset session**.

### 4.4. Slash commands

`/`-prefixed text in the composer opens
`SlashCommandPaletteView`. Built-in commands:

| Command | What it does |
|---|---|
| `/mode plan` / `/mode build` | Flip the current chat's `AgentMode`. |
| `/tools` | Open the Tools settings tab. |
| `/permissions` | Open the Permissions settings tab. |
| `/skill <name>` | Activate one of the agent's allowed skills inline. |
| `/theme` | Open the Theme settings tab. |
| `/clear` | Clear the current chat's draft (does not delete history). |
| `/help` | List the available commands. |

Custom slash commands per-project / per-agent are on the roadmap.

### 4.5. Skills

A **Skill** is a reusable bundle of (system prompt addendum,
suggested tool allowlist, optional default mode). Built-in
skills declared at stable UUIDs in `BuiltInSkills`; the agent
edit sheet's "Allowed skills" multi-pick stores `[UUID]` so the
agent can fall back to "no skill restriction" by leaving the
list empty.

Custom skills are CRUD-able under **Settings → Skills**.

---

## 5. The macOS app: tab-by-tab Settings reference

Reachable via `Cmd+,`. Most changes apply on the next Send; the
exceptions are noted per tab.

| Tab | Controls |
|---|---|
| **Generation** | Temperature (slider 0.5–1.0, default 0.7), top-K (0 = disabled), top-P, max-tokens, thinking mode. Overridden when an agent is attached to the chat. |
| **Loading** | Loader strategy override (`auto`/`preload`/`mmap`/`streaming`), force-load toggle, converter binary path. Apply at the next model load. |
| **Model Config** | Every field of `ModelConfig`. Writes to `~/Library/Application Support/<app>/config-overrides.json`. The loader honours `max_seq_len` / `max_batch_size` on the next load. |
| **Agents** | CRUD for agent presets — name, summary, system prompt, **Plan/Build mode**, thinking mode, **allowed skills**, MCP tool allowlist, sampling defaults, icon + tint. |
| **Tools** | Read-only inventory of every native tool the registry knows + a Plan/Build availability matrix. |
| **Permissions** | Per-(tool, category) `ask` / `alwaysAllow` / `alwaysDeny` defaults consulted before every dispatch. Reset session grants. |
| **Skills** | Manage the skill library — built-in entries are read-only; custom ones get a full edit sheet. |
| **Theme** | Appearance (light / dark / system), accent + bubble tints, custom theme import. |
| **Keybindings** | Read-only list of every action + its shortcut + a Reset-to-defaults button. Inline rebind UI is on the roadmap. |
| **Documents** | Import single text files; each is tokenised once against the loaded local model's tokenizer. |
| **Projects** | Group documents into projects for one-shot context injection in the chat (local only). |
| **MCP** | Register Model Context Protocol servers (stdio). Import from Claude Desktop config JSON. Live status per server. |
| **API Keys** | OpenRouter API key (Keychain). Save / Test / Delete. |
| **Storage** | Conversation history location, size on disk, Reveal in Finder, Clear all. |

---

## 6. Resume after crash

The local path snapshots a `PendingTurn` on every Send (prompt
tokens + sampled-so-far ids + mode). If the app dies mid-stream,
the chat shows a "Resume" banner above the composer on next open.
Click it to re-feed the snapshot through the model and continue
exactly where you left off.

Remote chats don't snapshot — re-issuing the HTTP request is
cheap, just re-send the message.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `MTLLibraryErrorDomain code 6: no default library` | metallib not built | re-run `swift build -c release`; check `find .build -name '*.metallib'`. |
| `Permission denied` on `build_metallib.sh` | not executable | `chmod +x Plugins/MetalLibPlugin/build_metallib.sh`. |
| `no .safetensors files in …` | wrong path | check the directory holds `*.safetensors` (not LFS pointers). |
| `all safetensors files … were LFS pointers` | git clone without LFS payload | re-download with `huggingface-cli download`. |
| `No space left on device` mid-convert | output volume full | move `--save-path`, or use `--target-dtype keep`/`int4` for smaller output. |
| Local first-token latency huge | cold SSD cache + streaming strategy | normal on first run; subsequent runs warm up. |
| Local model loops a single token | `--temperature 0` | switch to `--temperature 0.7`. The GUI clamps the slider. |
| OpenRouter chat says "API key not configured" | Keychain empty | Settings → API Keys → paste + Save. |
| OpenRouter 401 / 403 | invalid / expired key, out of credit | Settings → API Keys → Test. Check OpenRouter dashboard. |
| OpenRouter tool calls do nothing | MCP server offline | Settings → MCP → check footer says "Connected · N tools"; Reconnect if not. |
| Reasoning disclosure empty on remote | model doesn't emit `reasoning_content` | only DeepSeek-R1 / o-series do; others include thinking inline in `content` (Claude extended-thinking) or not at all. |
| `precompiled file '…' was compiled with module cache path` | stale Xcode caches after moving the project | `rm -rf .build && swift package clean && swift build`. For Xcode also wipe `~/Library/Developer/Xcode/DerivedData/DeepSeekV4Pro-*`. |
| `Cannot find type 'MCPClientPool'` after `git pull` | `.xcodeproj` is out of date | re-run `./Tools/generate-xcodeproj.sh`. |
| `No Metal device` on launch | Intel Mac | local inference requires Apple Silicon. Remote (OpenRouter) works on any Mac with macOS 14. |

When something fails, the most useful output is the last ~20 lines
before the crash plus the command you ran. For local-only build
issues see [`DEVELOPING.md`](DEVELOPING.md#common-pitfalls).
