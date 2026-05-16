# DeepSeek V4 on macOS

Swift + Metal port of the [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
Mixture-of-Experts transformer for Apple Silicon, plus a native SwiftUI
desktop client that doubles as a generic chat surface for any
OpenRouter-hosted model.

The desktop app supports:

- **Local inference** on V4-Flash weights (FP8 + FP4 native), streaming the
  ~142 GB checkpoint on 16 GB Macs through a per-layer rotating buffer.
- **Remote inference** through OpenRouter — any OpenAI-compatible model
  (Claude, GPT, DeepSeek-R1, Llama 3, etc.) with one API key.
- **MCP (Model Context Protocol) servers** as tool providers, identical
  config to Claude Desktop.
- **Agent presets**: system prompt + tool allowlist + sampling defaults +
  thinking mode pinned per chat. Agents can delegate sub-tasks to other
  agents (bounded nesting, cycle prevention).
- **Projects**: pre-tokenised codebases / document collections you can
  attach to a chat so the first turn already carries the context.

> **Experimental.** V4-Pro itself (1.6T parameters, ~800 GB at FP4) does not
> fit in any Mac's unified memory; the realistic on-device target is
> **DeepSeek-V4-Flash** (284B / 13B activated). Remote inference through
> OpenRouter is the answer when you want Claude, GPT, or other models you
> can't run locally.

🇮🇹 [Versione italiana](README.it.md) · 🏗 [Architecture deep-dive](docs/ARCHITECTURE.md)
· 🧪 [Testing](docs/TESTING.md) · 🛠 [Developing](docs/DEVELOPING.md)

---

## System requirements

| What | Local inference | Remote (OpenRouter only) |
|---|---|---|
| **CPU/GPU** | Apple Silicon (M1, M2, M3, M4…) | Any Apple Silicon Mac |
| **macOS** | 14.0 Sonoma | 14.0 Sonoma |
| **RAM (unified)** | 16 GB (V4-Flash, streaming) — 64+ GB recommended | 8 GB |
| **Disk** | 150 GB free for V4-Flash weights | A few MB for the app |
| **Tooling** | Swift 5.10 / Xcode 15+ | Swift 5.10 / Xcode 15+ |
| **Network** | optional | required |

The loader picks a local-inference strategy automatically based on
available RAM:

| Available RAM | Strategy | Behaviour |
|---|---|---|
| ≥ 192 GB | `preload` | Whole model resident, fastest |
| 32–192 GB | `mmap` | OS pages in on demand, fast after warm-up |
| 16–32 GB | `streaming` | One layer's shard at a time, slower first token |

Intel Macs are not supported — the Metal pipelines require
`bfloat`-capable hardware and `Sources/DeepSeekKit/Device.swift` will
refuse to initialise.

---

## 1. Get the project

```bash
git clone https://github.com/giorgiooppo/DeepSeek-V4-Pro-MacOS.git
cd DeepSeek-V4-Pro-MacOS
swift package resolve
```

The repo does not include the model weights, the tokenizer, or any API
key — those are gitignored / Keychain-stored.

## 2. Build

### CLI only

```bash
swift build -c release
```

Produces:

- `.build/release/deepseek` — local-inference CLI
- `.build/release/converter` — offline weight transcoder

The CLI talks only to local checkpoints — there's no OpenRouter dispatch
in `deepseek` itself; remote support lives only in the GUI.

### GUI app (Xcode)

```bash
brew install xcodegen        # one-time
./Tools/generate-xcodeproj.sh
open DeepSeekV4Pro.xcworkspace
```

Pick the **`DeepSeekApp`** scheme (the workspace exposes the SPM
executable target `DeepSeekUI` too — pick the app target, not that
one) and press ⌘R.

The app starts on the chat surface immediately. No model is required
to launch — you can browse history, edit agents / projects / MCP
servers, and queue a draft message before any backend is loaded.

---

## 3. Use a local model

### Download weights

The recommended on-device checkpoint is **DeepSeek-V4-Flash** in its
native HuggingFace layout (FP8 attention + FP4 experts). The Swift
loader reads that layout directly — no conversion step.

```bash
pip install --upgrade huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
    --local-dir ~/Downloads/V4-Flash-HF
```

The destination folder must contain:

- `config.json`, `generation_config.json`
- `tokenizer.json`, `tokenizer_config.json`
- `model.safetensors.index.json`
- 46 `model-NNNNN-of-NNNNN.safetensors` shards (~142 GB total)

The companion `converter` binary is **only** needed if you want to
transcode the checkpoint to BF16 / INT8 / INT4 / INT2 for smaller-disk
variants — see [`docs/USAGE.md`](docs/USAGE.md).

### Load in the GUI

1. Open the chat toolbar's **Model** menu (cpu icon, leftmost).
2. **Choose model folder…** → pick `~/Downloads/V4-Flash-HF`.
3. A banner above the composer shows `Loading <name>… <gb> GB across
   <n> shards · strategy: <preload|mmap|streaming>`. Wait for it to
   disappear.
4. The model picker label flips to the folder name. Send is enabled.
5. The folder is remembered under **Recent** — next launch auto-loads
   the last one.

Use **Unload current model** from the same menu to release RAM
without quitting. The chat history is independent of the loaded model.

### Run from the CLI

```bash
./.build/release/deepseek ~/Downloads/V4-Flash-HF \
    "What is the capital of France?" \
    --mode chat --max-tokens 50 --temperature 0.7
```

See [§ CLI reference](#cli-reference) below.

---

## 4. Use a remote model through OpenRouter

OpenRouter is an OpenAI-compatible API gateway that routes to ~300
models from Anthropic, OpenAI, DeepSeek, Meta, Mistral, and others
with a single key.

### One-time: add your API key

1. Get a key at <https://openrouter.ai/keys>.
2. Open the app's **Settings → API Keys** tab.
3. Paste it into the OpenRouter SecureField. **Save**.
4. (Optional) Click **Test** — hits `/auth/key` and shows a green
   "Key accepted" if the credentials work.

The key is stored in the macOS **Keychain** (service
`com.deepseek.v4pro`, account `openrouter.apiKey`). It is never
written to a plist or anywhere else readable from user space.

### Pick a model

1. Toolbar **Model** menu → **Add OpenRouter model…**.
2. The sheet loads OpenRouter's full catalog (cached for 24 h locally
   under `Application Support/.../openrouter-catalog.json`). Search by
   provider / name / slug; rows show context length, per-token
   pricing, short description.
3. Click a row → `ModelState` validates the key + flips the chat to
   that endpoint.
4. The model now appears under **Recent** alongside any local
   folders, and the toolbar label shows the slug (e.g.
   `claude-3.5-sonnet`).

### Send and watch costs

- Reply streams through SSE just like a local generation.
- DeepSeek-R1 and o-series reasoning is captured in
  `reasoning_content` and rendered through the same brain-icon
  disclosure as local `<think>` blocks.
- The **ThroughputBar** under the bubble shows `Turn cost: $0.0042`
  for the most recent turn.
- A separate banner above the composer shows `Chat total: $0.013`
  cumulative across the conversation, persisted across launches.

### What works on remote

- MCP tools (see § Agents & tools): exposed automatically as OpenAI
  `tools` array; tool-call loop fires HTTP round-trips up to 8 iters.
- Agent presets (system prompt + sampling defaults + tool allowlist):
  applied to the request body.
- Reasoning mode (`high` / `max`): translated to OpenRouter's
  `reasoning: {effort}` hint. Providers that don't support it (most
  non-R1/o-series) silently ignore.
- Thinking-mode picker (above composer): selection respected per turn.

### What doesn't work yet on remote

- **Cross-agent delegation** (`__delegate_to_agent`): not exposed on
  remote chats. Local agents handle this fully — remote variants would
  need a remote sub-agent loop that hasn't been written yet.
- **Crash recovery**: remote turns don't snapshot a `pendingTurn`. If
  the app dies mid-stream the turn is lost (re-send the message).
- **Prompt caching** (Anthropic / OpenAI prompt-cache discounts via
  OpenRouter): not yet implemented.

---

## 5. The macOS app

### Toolbar pickers

Left-to-right, four menus:

| Picker | Icon | Purpose |
|---|---|---|
| **Model** | cpu / internal drive / cloud | Switch backends. Local folder, remote OpenRouter model, browse, unload. |
| **Agent** | configured per agent | Attach an `AgentConfig` preset to this chat. None / list. |
| **Project** | folder | Attach a pre-tokenised project so the first turn carries that context. None / list. |
| **Convert** | wand | Opens the offline weight-quantisation sheet. |

Each picker reflects the active conversation; switching conversations
in the sidebar flips the labels.

### Chat surface

Pinned above the composer, top → bottom:

1. **Model-state banner**: hidden when ready; otherwise "Loading
   …" / "No model" / "Could not load …" with retry / force-load.
2. **Cumulative-cost banner**: `Chat total: $X.XX` for remote chats
   that have billed anything. Hidden for local chats.
3. **Live delegation chain**: stacked cards for every in-flight sub-
   agent (each with its icon + task + streaming reply, indented by
   depth). Empty stack collapses.
4. **Resume banner**: when a prior generation died mid-stream the
   `pendingTurn` snapshot offers a one-click resume.
5. **Thinking picker**: segmented control with `No think / High /
   Max`. Disabled with a lock hint when an agent overrides the mode.
6. **Composer**: TextField (Cmd+↩ to send) + Send/Stop button. Send
   disables when no model is loaded.

While streaming, the in-progress assistant bubble shows a blinking
caret. The ThroughputBar below it ticks `tok/min` and (for remote)
`Turn cost`. Reasoning content is folded into a brain-icon
disclosure; tool calls + outputs are folded into a wrench-icon
disclosure under the bubble.

### Sidebar

Conversation list with date stamp. Right-click → Delete. Cmd+N
creates a new chat under the current model. A small spinner marks
the chat that's actively generating.

---

## 6. Agents and tools

### Agents (Settings → Agents)

An **Agent** is a preset that bundles:

- `name`, `summary`, icon + tint;
- system prompt injected before every turn of the chat it's attached to;
- sampling defaults (`temperature`, `topP`, `topK`, `repPenalty`,
  `maxTokens`, `defaultMode`) that override the Generation-tab sliders;
- MCP tool allowlist (`nil` = all tools, empty = none, explicit set =
  whitelist of qualified `<server>__<tool>` names).

Attach one to a chat from the toolbar's Agent picker; detach with
"None". When attached, the thinking picker locks to the agent's
`defaultMode`, and the chat's sampling settings come from the agent
even if you tweak the global sliders.

Define agents under **Settings → Agents**: master-detail with an
edit sheet for everything above plus a tool-policy segmented control.

### Sub-agent delegation

When two or more agents are registered, the agent attached to a chat
gets a synthetic tool called `__delegate_to_agent` whose schema
includes a roster of every other agent. The model can call it with
`{ agent_name, task }`; the host runs the named agent in isolation
through a snapshot/restore of the KV cache and returns its final
reply as the tool output. Bounds:

- **Nesting cap**: up to 3 levels (`host → sub → sub-sub → sub-sub-sub`).
  The schema isn't injected at the cap so the model can't even try to
  delegate further.
- **Cycle prevention**: an agent already in the active call stack is
  refused with a structured error string. The model can self-correct
  by picking a different agent.
- **Cache preservation**: each level's KV cache is snapshotted before
  the sub-agent runs and restored after, so the host doesn't pay a
  cold re-prefill on return.

The chat surfaces the live chain (with each level's streaming buffer)
in a card pinned above the composer. Delegation is local-model only
for now.

### MCP servers (Settings → MCP)

Configure stdio-based MCP servers exactly like Claude Desktop:
command + args + env. Import from a Claude Desktop config JSON
(`mcpServers: { … }`) with one click.

Enabled servers spawn at app launch through the user's shell PATH
(homebrew, fnm, pyenv, …) plus an extended fallback so launchd's
stripped environment doesn't break tool discovery. Each row's status
footer shows connection state + the live tool list. Reconnect button
is inline.

Tools are exposed to the active chat through whichever backend it's
on (local DSML tool blocks, or OpenAI `tools` array for remote).

### Projects (Settings → Projects) & Documents (Settings → Documents)

A **Document** is a single text file ingested once and pre-tokenised
against the active model's tokenizer. A **Project** is a named
collection of documents.

Attach a project to a chat from the toolbar's Project picker. On the
first turn of that chat the project's files are spliced into the
prompt with the native repo / file delimiter tokens (`<｜begin▁of▁repo▁name｜>`
etc.), so the model treats them as code-aware context rather than
free-form text. Local-model only.

---

## 7. Preferences

All tabs are reachable via `Cmd+,`. Changes take effect on the next
Send (or, for `Model Config`, the next model load).

| Tab | Controls |
|---|---|
| **Generation** | Temperature (slider 0.5–1.0, default 0.7), top-K (0 = disabled), top-P, max-tokens, thinking mode. Overridden when an agent is attached. |
| **Loading** | Loader strategy override, force-load toggle, converter binary path. |
| **Model Config** | Every field of `ModelConfig`. Writes to `~/Library/Application Support/<app>/config-overrides.json`. |
| **Agents** | CRUD for agent presets — see § Agents. |
| **Documents** | Import single documents (tokenises against the loaded local model). |
| **Projects** | Group documents into projects for one-shot context injection. |
| **MCP** | Register MCP servers — see § MCP. |
| **API Keys** | OpenRouter API key (Keychain). Save / Test / Delete. |
| **Storage** | Conversation history location, size on disk, Reveal / Clear all. |

---

## 8. CLI reference

The CLI is local-only (no OpenRouter dispatch).

```
deepseek <model-dir> "<prompt>" [options]
```

Two positionals, the second optional only in diagnostic modes.

### Generation flags

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--mode` | `raw` \| `chat` | `chat` | `raw` prepends only BOS; `chat` applies the V4 chat template. |
| `--thinking` | `off` \| `high` \| `max` | `off` | Chat-mode reasoning budget. `off` appends `</think>` and the model answers directly; `high` appends `<think>`; `max` also prepends the REASONING_EFFORT_MAX system block. |
| `--temperature` | float | `1.0` | Sampling temperature. **Set to `0.7`** — see Recommended values. |
| `--max-tokens` | int | `32` | Maximum tokens to generate. |

### Loader / memory flags

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--load-strategy` | `auto` \| `preload` \| `mmap` \| `streaming` | `auto` | Force a specific loader path. |
| `--force-load` | flag | off | Bypass the conservative RAM safety checks. |
| `--max-seq-len` | int | from `config.json` | Override KV-cache rows per layer. Lower = less RAM, shorter context. |
| `--max-batch-size` | int | from `config.json` | Override the batch dimension of the KV cache. |

### Diagnostic modes

| Flag | What it does |
|---|---|
| `--print-config` | Loads `config.json`, prints the resolved `ModelConfig`, exits. |
| `--trace-norms` | Prints L2 norm + min/max/mean + NaN/Inf counters of the residual stream. |
| `--list-tensors [PREFIX]` | Lists every tensor name in the checkpoint, optionally filtered. |
| `--dump-tensor NAME[:row=R][:cols=A..B]` | Dequantises one row slice and prints the values. |

### Recommended values

- **`--temperature 0.7`**. V4-Flash's MoE routing under greedy argmax
  falls into self-reinforcing loops (`好的好的好的…`). Values around
  0.6–0.9 give the most coherent samples. The GUI clamps the slider to
  `[0.5, 1.0]` for the same reason.
- **`--mode chat --thinking off`** for short Q&A; `--thinking high`
  for problems where the model should "think out loud".

### Example

```bash
./.build/release/deepseek ~/Downloads/V4-Flash-HF \
    "Explain nuclear fusion in two sentences." \
    --mode chat --thinking off \
    --temperature 0.7 --max-tokens 256
```

---

## 9. Troubleshooting

**First token takes minutes on a local model.**
Expected under `streaming` on a 16 GB Mac — each layer's ~3 GB shard
has to read from disk before processing. Subsequent tokens are much
faster (the rotating slot stays warm).

**OpenRouter chat fails with "API key not configured".**
Settings → API Keys → paste and Save. The picker also shows an inline
warning when no key is configured.

**OpenRouter returns 401 / 403.**
Key is invalid, expired, or out of credits. Hit Test in the API Keys
tab; check the OpenRouter dashboard for usage limits.

**Tool calls don't execute on remote.**
Check that the MCP server's status footer shows `Connected · N tools`
in Settings → MCP. The remote path uses the same pool as local; if
the server is offline the model gets back a structured error.

**Reasoning content doesn't appear on remote.**
Only models like DeepSeek-R1 and o-series emit `reasoning_content`.
Other models include reasoning inline in `content` (Claude
extended-thinking) or not at all (Llama). The brain-icon disclosure
only renders when the field is populated.

**Build error: `precompiled file '…' was compiled with module cache
path '…'`.**
Stale Xcode caches, usually after moving the project folder. Wipe:

```bash
rm -rf .build
swift package clean
swift build
```

For Xcode, also clear `~/Library/Developer/Xcode/DerivedData/DeepSeekV4Pro-*`.

**Local model just loops a single token.**
You're sampling with `--temperature 0`. V4-Flash needs stochastic
sampling — pass `--temperature 0.7`. The GUI clamps this.

**"No Metal device" / Intel Mac.**
Apple Silicon is required for local inference. Remote (OpenRouter)
still works on any Mac that meets the macOS 14 minimum.

**Out of memory at local load.**
Try `--load-strategy streaming` to force the per-layer rotating
loader, or `--max-seq-len 2048 --max-batch-size 1` to shrink the KV
cache.

**"Cannot find type 'MCPClientPool'" / similar after a git pull.**
The Xcode project is generated from `project.yml`. Regenerate after
adding new files:

```bash
./Tools/generate-xcodeproj.sh
```

`swift build` does not need regeneration — SPM picks up new files
automatically.

---

## 10. License and credits

The Swift code in this repository is MIT-licensed (see [`LICENSE`](LICENSE)).
The DeepSeek model weights and the Python reference implementation in
`Reference/inference/` belong to DeepSeek — see their [Hugging Face
card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) for license
terms.

OpenRouter is a separate third-party service; your usage is governed
by their terms.

If you want to understand how the port works under the hood — kernel
mapping, residual amplification, streaming pool design, MoE dispatch
— read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For the
contributor workflow see [`docs/DEVELOPING.md`](docs/DEVELOPING.md),
and for ready-made prompts / recipes see [`docs/EXAMPLES.md`](docs/EXAMPLES.md).
