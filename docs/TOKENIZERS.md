# Tokenizers and chat templates

Everything the engine does between a user-typed string and the
`[[Int]]` of token ids that `Transformer.forward` consumes. Three
tokenizer formats, a chat-template dispatcher, the native DSV4
template, and a hand-rolled Jinja2 subset for everything else.

Companion docs:

- [`MODEL.md`](MODEL.md) — what the engine does with the ids.
- [`MODULES.md`](MODULES.md) — per-file index of `Sources/`.
- [`USAGE.md`](USAGE.md) — operational walkthrough (download +
  point the loader at the right directory).
- [`GAP-ANALYSIS-LLAMACPP.md`](GAP-ANALYSIS-LLAMACPP.md) — how the
  tokenizer + template dispatcher compares to llama.cpp.

> 🇮🇹 La versione italiana è [`TOKENIZERS.it.md`](TOKENIZERS.it.md).

---

## 1. The subsystem

```
~/Downloads/Model/                       on-disk
├── tokenizer.json                       (HF: BPE / WordPiece / Unigram)
├── tokenizer_config.json                (HF: chat_template, model_type)
├── tokenizer.model                      (SentencePiece protobuf — only for SP models)
└── …
        │
        ▼
TokenizerLoader.load(tokenizerDir:)
        │
        ├─ pick tokenizer
        │     · tokenizer.json model.type == "BPE"        → BPETokenizer
        │     · tokenizer.json model.type == "WordPiece"  → WordPieceTokenizer
        │     · tokenizer.json model.type == "Unigram"
        │       OR sibling *.model                        → SentencePieceTokenizer
        │
        └─ pick chat template
              · DSV4 vocab signature (or model_type contains "deepseek")
                                                          → DSV4Template (uses EncodingDSV4)
              · tokenizer_config.json.chat_template       → JinjaChatTemplate (Jinja2 subset)
              · neither                                   → throw .unsupportedType
        ▼
LoadedTokenizer { tokenizer, chatTemplate, isDSV4 }
```

Three independent pieces:

1. **Tokenizer**: text → `[Int]` ids and back. Three implementations,
   one for each format we encounter.
2. **Chat template**: list of messages → prompt string. Two
   implementations: the native DSV4 path and a Jinja2-subset driver
   for everything else.
3. **`TokenizerLoader.load`**: the dispatcher that inspects a
   directory and instantiates the right pair.

The chat surface only ever sees `LoadedTokenizer`. The CLI uses both
the tokenizer (for raw mode) and the template (for chat mode).

---

## 2. The `Tokenizer` protocol

`Sources/DeepSeekKit/Tokenizer.swift:6`:

```swift
public protocol Tokenizer {
    func encode(_ text: String) -> [Int]
    func decode(_ ids: [Int]) -> String
    var bosId: Int? { get }
    var eosId: Int? { get }
    var stopTokenIds: Set<Int> { get }
}
```

Five members. The first two are obvious. The last three drive the
generation loop:

- `bosId` is prepended to the prompt in raw mode; chat templates
  embed it explicitly when they want.
- `eosId` is the canonical "stop" token.
- `stopTokenIds` is the *set* of every id whose sampling terminates
  generation. It always includes `eosId` plus any other terminal
  marker the vocab defines (V4's `<|EOT|>`, Llama's `<|eot_id|>`).

The decode loop checks `stopTokenIds`, not just `eosId`. Earlier
versions tested only against `eosId` and V4 chat generations ran past
the assistant's end-of-turn marker, where the LM head's residual
ranking collapsed into looped filler tokens. That bug is fixed by
treating the stop check as a set membership.

---

## 3. BPE (DeepSeek-V4 and friends)

`Sources/DeepSeekKit/BPETokenizer.swift:26`. Byte-level BPE compatible
with HuggingFace `tokenizer.json` files produced by the `tokenizers`
library. This is the format DeepSeek-V4 ships and the format the
Project / Document indexing path assumes.

### Algorithm

1. **Pre-tokenisation**: split the input on the GPT-2 / cl100k regex
   (or whatever pattern `model.pre_tokenizer.pattern.Regex` provides).
   Each match becomes one "word" that BPE operates on independently.
2. **Byte-to-unicode encoding**: convert the UTF-8 bytes of the word
   through the GPT-2 byte-to-unicode map. Bytes that would be control
   characters or whitespace get rotated into the printable Latin-1
   range so the BPE merges work on display-safe characters.
3. **Greedy lowest-rank BPE**: starting from per-character symbols,
   look at every adjacent pair, pick the one with the lowest merge
   rank, fuse it, and repeat until no more applicable merges remain.
4. **Vocab lookup**: each final symbol gets `vocab[symbol] → Int`.

Special tokens (`added_tokens` in tokenizer.json) bypass the pipeline
entirely — the encoder splits the input on them first and emits the
exact id straight through.

### Decoding

Symbol-by-symbol `invVocab[id]` lookup, then map each byte-encoded
character back through `unicodeToByte`, then UTF-8 decode. Special
tokens decode to their text content as-is.

### Limitations vs the full HF spec

- No normalisation step (DeepSeek's `tokenizer.json` has `normalizer:
  null`; if a model needs NFC/NFD/NMT before BPE, this implementation
  doesn't apply it).
- No truncation/padding (caller-managed).
- Regex split uses `NSRegularExpression` with the most-common
  ByteLevel pattern; if a tokenizer specifies a different
  `pretokenizer.pattern.Regex`, we use that pattern verbatim — but
  any features `NSRegularExpression` doesn't support (lookbehind
  variants, Unicode property escapes beyond the basics) won't parse.

### Concurrency

`BPETokenizer` is marked `@unchecked Sendable`. Every stored property
is `let`, and `NSRegularExpression` is documented thread-safe by
Apple. The vocab-pruning tool calls `encode(_:)` from multiple threads
when `--concurrency` is > 1.

---

## 4. SentencePiece (Llama / Mistral / Qwen / Gemma)

`Sources/DeepSeekKit/SentencePieceTokenizer.swift:14`. Minimal
implementation that reads the binary `.model` protobuf file ad-hoc —
only the tags we need are decoded, the rest is skipped, so we ship
without a dependency on `swift-protobuf`.

### Algorithm

Unigram language model with **Viterbi decoding** over a lattice:

1. **Whitespace normalisation**: every space becomes `▁` (U+2581,
   "lower one eighth block"). This is the SentencePiece convention —
   space-prefixed tokens preserve word boundaries without needing a
   pre-tokeniser.
2. **Lattice**: at every position `i` in the (normalised) input,
   enumerate every piece in the vocab that matches the prefix. Each
   piece is an edge with cost `-score(piece)` (lower-score = more
   likely).
3. **Viterbi**: dynamic programming over edges to find the
   minimum-cost path from start to end.
4. **Byte fallback**: characters with no matching piece in the vocab
   fall back to per-byte `<0xNN>` tokens (`type == 6` in the proto).
   Every SentencePiece model that ships with HF LLMs includes the
   256-byte BYTE table by default.

### Decoding

`pieces[id].text` lookup, byte-fallback for `<0xNN>` tokens (parsed
back to a byte and re-glued), then strip the leading `▁` and replace
remaining `▁`s with spaces.

### Limitations

- BPE-style SentencePiece (older Llama 1) isn't supported — only the
  Unigram variant. The proto schema is read field-by-field; an
  unknown wireType throws.
- No subword regularization (SentencePiece's nbest path is dropped
  for inference).

---

## 5. WordPiece (BERT-style, for embedding/classification models)

`Sources/DeepSeekKit/WordPieceTokenizer.swift:9`. BERT-style WordPiece
that reads the `vocab` field of a HuggingFace `tokenizer.json` whose
`model.type == "WordPiece"`.

### Algorithm

1. **Pre-tokenisation**: whitespace + punctuation split. Each
   "word" is encoded independently.
2. **Greedy longest-match**: for each word, start from the head and
   look for the longest match in the vocab. Subsequent matches use
   the `##` continuation prefix.
3. **Unknown handling**: words exceeding `maxInputCharsPerWord` (200)
   produce a single `[UNK]` id; words with no match for the first
   character also produce `[UNK]`.

### Limitations

The runtime engine doesn't ship a WordPiece-using model today.
WordPiece exists because the GGUF reader scope was widened to cover
BERT-class embedding models for retrieval / classification use cases.
The tokenizer is implemented and tested, but the *inference loop*
for these models hasn't landed — see `docs/GGUF.md` and the roadmap.

---

## 6. The dispatcher: `TokenizerLoader`

`Sources/DeepSeekKit/Tokenizer.swift:57`. One static entry point:
`TokenizerLoader.load(tokenizerDir: URL)` returns a
`LoadedTokenizer` with the tokenizer + chat template + a flag
distinguishing DSV4 from everything else.

Detection logic:

1. **Tokenizer**: read `tokenizer.json`'s `model.type`.
   - `"BPE"` → `BPETokenizer`.
   - `"WordPiece"` → `WordPieceTokenizer`.
   - `"Unigram"` or missing — look for a sibling `*.model` file →
     `SentencePieceTokenizer`. If neither (no `tokenizer.json` either),
     throw `missingFile`.
2. **Chat template**:
   - If `tokenizer_config.json.model_type` contains `"deepseek"`,
     OR the `tokenizer.json` `added_tokens` list contains
     `begin▁of▁sentence` → `DSV4Template()`.
   - Otherwise, if `tokenizer_config.json.chat_template` is
     non-empty → `JinjaChatTemplate(src)`.
   - Otherwise throw `unsupportedType`.

DSV4 detection wins over a generic Jinja template even when one is
present: `EncodingDSV4` wraps additional logic beyond pure message
rendering (the REASONING_EFFORT_MAX prompt, the tools block, the
`<｜tool▁outputs｜>` post-processing) that the Jinja-only path doesn't
reproduce. Future non-V4 DeepSeek models still pick up their own
`chat_template`.

### Legacy shim

`TokenizerLoader.load(from url: URL)` takes a path directly to a
`tokenizer.json` file and returns only the `Tokenizer` — no chat
template. Older callers (and the converter) use this when they don't
need rendering.

---

## 7. The `ChatTemplate` protocol

`Sources/DeepSeekKit/Encoding/ChatTemplate.swift:11`:

```swift
public protocol ChatTemplate: Sendable {
    func render(messages: [Message], options: ChatTemplateOptions) throws -> String
}
```

One method: render a list of `Message`s into the prompt string the
model expects. Two implementations:

- **`DSV4Template`** — wraps `EncodingDSV4.encodeMessages(...)`.
  Used for every DeepSeek-V4 checkpoint.
- **`JinjaChatTemplate`** — wraps the Jinja2 subset driver. Used
  when the loaded model's `tokenizer_config.json` carries a
  `chat_template` field (Llama / Mistral / Qwen / ChatML / any HF
  model with a template).

### `ChatTemplateOptions`

```swift
public struct ChatTemplateOptions: Sendable {
    public var addGenerationPrompt: Bool   // append the trailing role marker
    public var thinkingMode: ThinkingMode  // DSV4-specific
    public var toolSchemasJSON: String?    // DSV4-specific (DSML format)
    public var tools: [JSONValue]?         // Jinja-specific (OpenAI format)
    public var bosToken: String            // some Jinja templates need this
    public var eosToken: String            // some Jinja templates need this
}
```

The split between `toolSchemasJSON` and `tools` is intentional: DSV4
expects pre-serialised JSON the host built once and pastes verbatim
into the `## Tools` section, while Jinja templates expect the OpenAI
shape (an array of objects with `name` / `description` /
`parameters`).

### `ChatTemplateError`

Three cases: `unsupportedFeature(msg)` (a Jinja construct we don't
support), `templateRaise(msg)` (a template called
`raise_exception(...)`), `parseFailure(msg)` (lexer / parser broke).
All bubble back to the caller, which translates them to a user-facing
error message.

---

## 8. The DSV4 template

`Sources/DeepSeekKit/Encoding/EncodingDSV4.swift` ports
`Reference/encoding/encoding_dsv4.py`. The high-level shape:

```
<｜begin▁of▁sentence｜>{system_content}
<｜User｜>{user_content_1}<｜Assistant｜>[<think>{reasoning}</think>]{assistant_content_1}
[<｜DSML｜tool_calls>...</｜DSML｜tool_calls>]
<｜end▁of▁sentence｜>
[<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>{name}<｜tool▁sep｜>{body}<｜tool▁output▁end｜>...<｜tool▁outputs▁end｜>]
<｜User｜>{user_content_2}<｜Assistant｜>[<think>|</think>]
```

The trailing `[<think>|</think>]` is the part the model fills in. The
encoder emits one of two markers depending on the thinking mode:

- `chat` (no-think): emit `</think>` (telling the model "no
  reasoning, here's the response").
- `high` / `max`: emit `<think>` (telling the model "start your
  reasoning"). The model's first generated tokens go inside the
  block; the `parseCompletion` path extracts them back into
  `Message.reasoningContent`.

### Special tokens

The DSV4 vocab includes ~25 markers used by the template. The full
list (`Sources/DeepSeekKit/Encoding/EncodingDSV4.swift:9-44`):

```
<｜begin▁of▁sentence｜>     bosToken
<｜end▁of▁sentence｜>       eosToken
<｜User｜>                  userToken
<｜Assistant｜>             assistantToken
<think>, </think>           thinkOpen, thinkClose

# Project / repo delimiters (real added_tokens 128815-820)
<｜begin▁of▁repo▁name｜>  / <｜end▁of▁repo▁name｜>
<｜begin▁of▁file▁name｜>  / <｜end▁of▁file▁name｜>
<｜begin▁of▁file｜>        / <｜end▁of▁file｜>

# Tool-output delimiters (real added_tokens 128810-814)
<｜tool▁outputs▁begin｜> / <｜tool▁outputs▁end｜>
<｜tool▁output▁begin｜>  / <｜tool▁output▁end｜>
<｜tool▁sep｜>
```

All of these are real tokens in the V4 vocab — the BPE pre-splits on
them so each emits exactly one id regardless of surrounding bytes.

### Thinking modes

`ThinkingMode` enum (`Sources/DeepSeekKit/Encoding/Message.swift:46`):

| Mode | Emitted trailing marker | System block prepended |
|---|---|---|
| `.chat` | `</think>` | — |
| `.high` | `<think>` | — |
| `.max` | `<think>` | `REASONING_EFFORT_MAX` |

`REASONING_EFFORT_MAX` is a hard-coded multi-line instruction
prepended to (or merged into) the first system message — see
`EncodingDSV4.swift:48-54` for the exact text.

### Tool schemas — the `TOOLS_TEMPLATE` block

When `toolSchemasJSON` is non-empty, the encoder prepends a `## Tools`
block to the first system message (or creates a system message if
there is none). The block instructs the model on the
`<｜DSML｜tool_calls>` invocation syntax and lists the schemas the host
made available.

The schemas come pre-serialised (as a single JSON string the host
already built, typically from `MCPClientPool.toolSchemasJSON(...)`
plus the native-tool registry's `availableSchemas(mode:)`). The
template doesn't try to re-render them — it just pastes.

### Tool calls (model output → host)

The encoder also handles the *outgoing* direction. After the
assistant content, it appends `<｜DSML｜tool_calls>` containing one
`<｜DSML｜invoke>` block per `ToolCall`:

```
<｜DSML｜tool_calls>
<｜DSML｜invoke name="filesystem__read_file">
<｜DSML｜parameter name="path" string="true">/etc/hosts</｜DSML｜parameter>
<｜DSML｜parameter name="limit" string="false">50</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
```

`string="true"` for raw string values; `string="false"` (with the
value as JSON) for everything else. XML-escaped inside the bodies.
The parsing pass (`parseCompletion`) is the inverse.

### Tool outputs (host → model)

When an assistant turn has `toolOutputs` populated, the encoder
appends a native-format block after that turn's EOS:

```
<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>filesystem__read_file<｜tool▁sep｜>...body...<｜tool▁output▁end｜><｜tool▁outputs▁end｜>
```

One `<｜tool▁output▁begin｜>…<｜tool▁output▁end｜>` per output, prefixed
with the corresponding tool name when known. Used as the splice point
in the tool-call loop: after the host executes a call, it writes the
result back into the prompt via this block, then re-fires
`generateForConversation`.

### Parsing model completions

`EncodingDSV4.parseCompletion(_:mode:)` is the inverse:

1. Strip a trailing `<｜end▁of▁sentence｜>` if present.
2. Extract an optional `<think>…</think>` block into
   `Message.reasoningContent`.
3. Extract an optional `<｜DSML｜tool_calls>…</｜DSML｜tool_calls>` block,
   parse each `<｜DSML｜invoke>`, rebuild the `ToolCall.args` JSON from
   the `<｜DSML｜parameter>` children.
4. Return the remainder (trimmed of whitespace) as `Message.content`.

The chat surface calls `parseCompletion` only at end-of-turn — during
streaming it shows the raw text as it arrives. The streaming UI knows
how to render an in-progress `<think>` block (the brain-icon
disclosure) by recognising the marker without waiting for the close
tag.

### What's not ported

The Python reference has more surface area; we've deliberately
deferred a few less-used pieces:

- `response_format_template` (the JSON-schema constrained-output
  block) — caller can prepend manually.
- `latest_reminder` — a once-per-turn "system reminder" the Python
  appends. Negligible value for the realistic chat flow.
- Task tokens (`<task>...</task>`) — niche, not in V4-Flash's
  training distribution as far as we can tell.

---

## 9. The Jinja2 subset

`Sources/DeepSeekKit/Encoding/JinjaTemplate.swift` is a hand-rolled
implementation of the subset of Jinja2 that HuggingFace chat
templates actually use. ~900 LOC. It's intentionally narrow:

| Feature | Supported |
|---|---|
| `{{ var }}` interpolation | ✅ |
| Chained `.field` / `[index]` access | ✅ |
| `{% if ... %}` / `{% elif %}` / `{% else %}` / `{% endif %}` | ✅ |
| `{% for x in xs %}` / `{% endfor %}` | ✅ |
| Filters (`trim`, `length`, `lower`, `upper`, `default`, `tojson`, `replace`, `string`) | ✅ |
| `raise_exception("...")` | ✅ (→ `ChatTemplateError.templateRaise`) |
| Comparison + boolean operators (`==`, `!=`, `in`, `and`, `or`, `not`) | ✅ |
| Arithmetic | ❌ |
| `{% set %}` | ❌ |
| `{% macro %}` / `{% include %}` / template inheritance | ❌ |
| `loop.first` / `loop.last` / `loop.index` | ✅ |

None of the omitted features appear in the chat templates we've
encountered. If one shows up, the error message
(`unsupportedFeature`) points directly at what to add.

### JinjaChatTemplate

`Sources/DeepSeekKit/Encoding/JinjaChatTemplate.swift` is the
adapter that maps the host's `[Message]` / `[ToolCall]` types into
the Jinja scope HuggingFace templates expect:

```
messages: list of dicts with keys
    role, content, tool_calls?, reasoning_content?
add_generation_prompt: bool
bos_token, eos_token: str
tools: list (OpenAI-style)
```

Each `ToolCall` is rendered as the OpenAI-shape `{ type: "function",
function: { name, arguments } }`. The Jinja template can then walk
`message.tool_calls` like Llama 3's official template does.

### When templates fail

The `parseFailure` error case carries the lexer / parser's diagnostic
(line number + offending token). The GUI surfaces this as "could not
load chat template" with the inner message — useful when a HF release
ships a template with a construct we haven't added yet.

---

## 10. Project / document tokenisation

For "attach a project to a chat" the engine pre-tokenises a directory
once and splices the result into the first user turn. The path is
local-model only (remote OpenRouter chats don't have access to the
native tokens).

`Sources/DeepSeekUI/Utility/ProjectIndexer.swift` walks the project's
files; for each file it builds:

```
<｜begin▁of▁repo▁name｜>{repo-name}<｜end▁of▁repo▁name｜>
<｜begin▁of▁file▁name｜>{path}<｜end▁of▁file▁name｜>
<｜begin▁of▁file｜>
{contents}
<｜end▁of▁file｜>
```

These are real tokens in the V4 vocab; the BPE pre-splits on them so
the model sees a clean structured boundary. The pre-tokenised id
array is stored under `Application Support/.../projects/<id>/`
keyed by `ModelFingerprint.of(modelDirPath:)` — re-importing is
required when the loaded model changes (different tokenizer = stale
ids).

`InferenceService.tokenizeFirstTurnWithProject(...)` is the entry
point the chat surface calls when sending the first message of a
chat with a project attached.

---

## 11. Vocab pruning (IT-only path)

A second tool, `Sources/DeepSeekVocabPruner/`, walks a corpus of
Italian text against the BPE tokenizer and finds the subset of vocab
ids that cover ≥ N% of the observed tokens. The intent: prune the
unused tail to shrink the embedding matrix.

This is an experimental feature, IT-language only, and not yet wired
into the runtime. Full docs in [`VOCAB-PRUNING.md`](VOCAB-PRUNING.md).

---

## 12. Cross-walk to the Python reference

| Python | Lines | Swift |
|---|---|---|
| `encode_messages` | 506–575 | `EncodingDSV4.encodeMessages` (`Sources/DeepSeekKit/Encoding/EncodingDSV4.swift:93`) |
| `parse_message_from_completion_text` | 687–744 | `EncodingDSV4.parseCompletion` (`:223`) |
| `tool_calls_template` / `tool_call_template` | 52–58 | `EncodingDSV4.encodeToolCalls` (`:185`) |
| `encode_arguments_to_dsml` | 139–180 | `EncodingDSV4.encodeArguments` (`:198`) |
| `REASONING_EFFORT_MAX` | 64–67 | `EncodingDSV4.reasoningEffortMax` (`:48`) |
| `TOOLS_TEMPLATE` | 70–95 | `EncodingDSV4.toolsBlock(toolSchemasJSON:)` (`:58`) |
| `response_format_template` | 49–51 | not ported — prepend in caller |
| Special token constants | 17–35 | top of `EncodingDSV4.swift` |
| `to_json` / `tools_from_openai_format` / etc. | 101–137 | not ported — caller passes pre-serialised JSON |

Reference test inputs (`Reference/encoding/tests/test_input_*.json`
plus `test_output_*.txt`) are golden tests for the encoder. The Swift
`Tests/DeepSeekKitTests/EncodingDSV4Tests.swift` consumes the subset
the port supports; the rest stay in the roadmap.

---

## 13. Source map

| Topic | File |
|---|---|
| Protocol + dispatcher | `Sources/DeepSeekKit/Tokenizer.swift` |
| Byte-level BPE | `Sources/DeepSeekKit/BPETokenizer.swift` |
| SentencePiece (Unigram + byte-fallback) | `Sources/DeepSeekKit/SentencePieceTokenizer.swift` |
| WordPiece (BERT-style) | `Sources/DeepSeekKit/WordPieceTokenizer.swift` |
| `Message` / `Role` / `ToolCall` / `ThinkingMode` | `Sources/DeepSeekKit/Encoding/Message.swift` |
| `ChatTemplate` protocol + options + errors | `Sources/DeepSeekKit/Encoding/ChatTemplate.swift` |
| DSV4 template (wrapper) | `Sources/DeepSeekKit/Encoding/DSV4Template.swift` |
| DSV4 encoder + parser | `Sources/DeepSeekKit/Encoding/EncodingDSV4.swift` |
| Jinja2 subset interpreter | `Sources/DeepSeekKit/Encoding/JinjaTemplate.swift` |
| Jinja chat template adapter | `Sources/DeepSeekKit/Encoding/JinjaChatTemplate.swift` |
| Project / Document tokenisation | `Sources/DeepSeekUI/Utility/ProjectIndexer.swift` |
| Tests | `Tests/DeepSeekKitTests/{BPETokenizerTests,EncodingDSV4Tests,JinjaTemplateTests}.swift` |
| Python reference | `Reference/encoding/encoding_dsv4.py` |

---

## 14. Limitations and deferred work

Tracked in `TODO.md` (sections "Encoding" and "Multi-format / GGUF /
chat-template dispatcher") and [`ROADMAP.md`](ROADMAP.md). At a glance:

- **WordPiece inference loop**: the tokenizer is implemented; the
  embedding-model forward pass that would consume those ids isn't.
- **Jinja `{% set %}` / `{% macro %}` / inheritance**: deferred —
  none of the templates in the wild we've encountered need them. The
  parser logs `unsupportedFeature` so adding support is a clean
  patch.
- **Constrained decoding (JSON schema)**: a separate concern that
  hangs off the sampler more than the tokenizer; tracked in
  `TODO.md §10.3`.
- **SentencePiece BPE-mode** (older Llama 1): not supported. Only
  Unigram + byte-fallback.
- **DSV4 `response_format_template` / `latest_reminder` / task tokens**:
  not ported; caller can prepend the literal strings if needed.
