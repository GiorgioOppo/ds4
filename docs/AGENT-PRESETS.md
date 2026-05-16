# Built-in agent presets

The app ships with seven curated `AgentConfig` presets, defined in
`Sources/DeepSeekUI/State/BuiltInAgents.swift` and seeded by
`AgentLibrary` the first time it launches against an empty
`agents.json`. After the first launch the file on disk belongs to
the user — the presets are never re-applied automatically, so any
edit, rename, or deletion sticks.

The presets are general-purpose ("generalisti di base"): they are
meant to cover the most common conversational, engineering, and
writing flows without committing to project-specific roles (a
"Metal Kernel Engineer" or "Quantization Expert" preset, for
example, would belong to a follow-up batch tailored to this repo's
internals).

## Design conventions

- **System prompts** are written in **English**. DeepSeek and the
  other supported backends follow English instructions more
  reliably, even when the user converses in another language. The
  prompt itself tells the agent to "match the user's language", so
  Italian replies come out naturally without baking the language
  into the prompt.
- **Summaries** (the one-liner shown in the sidebar and the agent
  picker) are in **Italian**, matching the local tone of
  `README.it.md` and the chat UI.
- **UUIDs** are stable: `4A4E000N-000N-4000-8000-00000000000N`
  where `N` identifies the preset slot (1 = Chat, 2 = Coder, ...,
  7 = Brainstormer). Stable IDs let downstream features — sharing,
  analytics, deep links — reference "the Coder preset" without
  depending on the localisable name.
- **Sampling defaults** are role-tuned. The Generation Settings
  sliders still override per-chat once the user touches them; the
  values below are the starting point for a fresh chat under each
  preset.
- **Tool / mode policy** uses two orthogonal knobs:
  - `agentMode` (`build` vs `plan`) decides which **native** tools
    the registry exposes. `plan` strips `mutating` and `dangerous`
    tools (write, edit, shell, apply_patch) before the model sees
    them; `build` exposes everything, gated by the permission
    policy.
  - `allowedToolNames` filters **MCP** tools by qualified name
    (`server__tool`). `nil` = expose every connected MCP tool,
    empty set = no MCP tools at all, explicit set = whitelist.

## The roster

### 1. Chat — Assistente generalista

| field | value |
| --- | --- |
| `agentMode` | `build` |
| `defaultMode` | `chat` (no thinking) |
| `allowedToolNames` | `nil` (all MCP tools) |
| `temperature` / `topP` | 0.7 / 0.95 |
| `maxTokens` | 4096 |
| icon / tint | `bubble.left.and.bubble.right` / blue |

**Italian summary:** "Assistente conversazionale generalista per
domande, chiarimenti e chiacchierate."

The default landing agent. Tools are technically available but the
prompt steers it away from filesystem and shell calls for casual
turns — those handoffs go to Coder or Researcher. Tone is friendly
and concise, matches the user's language, and is honest about
uncertainty. Low ceremony: short answers by default, sparse
Markdown, no unsolicited disclaimers.

### 2. Coder — Ingegnere software

| field | value |
| --- | --- |
| `agentMode` | `build` |
| `defaultMode` | `high` (think before coding) |
| `allowedToolNames` | `nil` |
| `temperature` / `topP` | 0.3 / 0.95 |
| `repetitionPenalty` | 1.05 |
| `maxTokens` | 8192 |
| icon / tint | `chevron.left.forwardslash.chevron.right` / purple |

**Italian summary:** "Ingegnere software con accesso al filesystem:
legge, modifica e testa il codice."

The full-power engineer. The prompt encodes the small-diff doctrine
(read before write, match local style, smallest change, no
speculative abstractions), the verification rule (run tests after
non-trivial edits, never claim success without proof), and the
safety perimeter (ask before destructive operations, never commit /
push without explicit consent, treat secrets as off-limits).
Low temperature plus a mild repetition penalty keeps Swift / Python
code stable across long generations.

### 3. Researcher — Esploratore read-only

| field | value |
| --- | --- |
| `agentMode` | `plan` |
| `defaultMode` | `high` |
| `allowedToolNames` | `nil` |
| `temperature` / `topP` | 0.4 / 0.9 |
| `maxTokens` | 6144 |
| icon / tint | `magnifyingglass` / teal |

**Italian summary:** "Ricercatore in modalità sola lettura: esplora
codice e web senza modificare."

Plan mode filters mutating tools out of the schema before the model
ever learns about them, so the agent literally cannot propose an
edit or a shell command — defence in depth on top of the prompt.
Available tools reduce to `read`, `glob`, `grep`, `websearch`,
`webfetch` (and any read-only MCP tools the user has connected).
The prompt enforces "headline answer first, then citations" and
explicit "I don't know" over speculation. Hands off to Coder when
the user wants execution.

### 4. Writer — Scrittore ed editor

| field | value |
| --- | --- |
| `agentMode` | `build` |
| `defaultMode` | `chat` |
| `allowedToolNames` | `nil` |
| `temperature` / `topP` | 0.8 / 0.95 |
| `frequencyPenalty` / `presencePenalty` | 0.2 / 0.1 |
| `maxTokens` | 8192 |
| icon / tint | `pencil.and.outline` / green |

**Italian summary:** "Scrittore ed editor per testi lunghi: bozze,
revisioni e riformulazioni."

Tuned for prose: slightly higher temperature for voice variety, a
small frequency / presence penalty to discourage the verbal tics
LLMs fall into in long-form generation. The prompt insists on
asking about audience / length / register before substantive
drafts, on preserving the author's voice during revisions, and on
delivering the revised text first followed by a short
"what changed" list (deliverable before commentary).

### 5. Translator — Traduttore professionale

| field | value |
| --- | --- |
| `agentMode` | `build` |
| `defaultMode` | `chat` |
| `allowedToolNames` | `[]` (no MCP tools) |
| `temperature` / `topP` | 0.2 / 0.9 |
| `maxTokens` | 4096 |
| icon / tint | `globe` / orange |

**Italian summary:** "Traduttore professionale tra italiano,
inglese e principali lingue europee."

Pure-LLM translation: empty MCP allowlist keeps the model from
reaching for external services it doesn't need. Default pair is
Italian ↔ English, with high confidence on the major European
languages. Low temperature keeps the rendering faithful. The
prompt mandates structural preservation (Markdown, code fences,
placeholders) and "translation only — no commentary, no source
echo" by default. Special-mode handling for literal /
word-for-word and full localisation requests.

### 6. Tutor — Tutor didattico

| field | value |
| --- | --- |
| `agentMode` | `plan` |
| `defaultMode` | `high` |
| `allowedToolNames` | `nil` |
| `temperature` / `topP` | 0.6 / 0.95 |
| `maxTokens` | 6144 |
| icon / tint | `graduationcap` / yellow |

**Italian summary:** "Tutor didattico: spiega concetti complessi
con esempi, intuizione e fonti."

Plan mode plus a teaching prompt: read and web-search to ground
answers, but no editing or shelling. The pedagogy is encoded in
the prompt — calibrate to the user's level first, prefer worked
examples over abstractions, build intuition before formalism, give
hints in increasing strength rather than the full answer up front.
Socratic style is rationed; some users want the answer, not a
question back.

### 7. Brainstormer — Compagno di ideazione

| field | value |
| --- | --- |
| `agentMode` | `build` |
| `defaultMode` | `chat` |
| `allowedToolNames` | `[]` |
| `temperature` / `topP` | 1.1 / 0.98 |
| `frequencyPenalty` / `presencePenalty` | 0.4 / 0.3 |
| `maxTokens` | 3072 |
| icon / tint | `lightbulb` / pink |

**Italian summary:** "Compagno di brainstorming: genera idee,
alternative e angolazioni divergenti."

High temperature and the strongest frequency / presence penalties
in the roster, to push divergent outputs and avoid the "ten
variations on the same idea" failure mode. Short `maxTokens` is
deliberate: ideation should produce many short ideas, not a few
long ones. The prompt names usable techniques (inversion,
constraint flips, pre-mortem, "yes, and...") and asks the model to
switch explicitly between divergent and convergent modes on
request.

## Adding a new preset

1. Add a `static var` to `BuiltInAgents` with a fresh UUID and a
   `createdAt` past the last existing offset (so the sidebar order
   stays stable for existing users).
2. Append it to the `defaults()` array.
3. Document it here — keep the table format consistent so the
   roster reads as a single reference.
4. Existing users will **not** see the new preset automatically —
   `AgentLibrary` seeds only when `agents.json` is missing. Ship
   new presets through a migration if you need them to land in
   existing installs, or rely on a "restore defaults" UI affordance
   (not implemented yet — see the open work in `TODO.md`).

## Related docs

- `docs/AGENT-MODES.md` — the build / plan mode mechanics and the
  permission policy that gates tool calls.
- `Sources/DeepSeekUI/State/AgentLibrary.swift` — runtime store,
  persistence, and the seeding logic that drops these presets in
  on first launch.
- `Sources/DeepSeekUI/Views/Agents/AgentEditSheet.swift` — the
  editor sheet users hit when they tweak a preset.
