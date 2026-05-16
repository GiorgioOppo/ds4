# Agent modes

Every agent has a coarse operating mode: **build** (the historical
behaviour) or **plan** (read-only stance). The mode is stored on
`AgentConfig.agentMode` and is honoured by the native tool registry
and by the slash-command palette.

## What the modes do

### Build

- Every registered tool is eligible.
- Mutating, dangerous, and network tools go through the permission
  policy. The first call per session per `(tool, category)` opens
  the `PermissionPromptView` sheet; the user can pick **Deny**,
  **Allow once**, or **Always allow**.
- Persistent defaults from `PermissionStore` are consulted before the
  prompt (`alwaysAllow` skips the sheet; `alwaysDeny` returns
  `denied` immediately without prompting).

### Plan

- `.mutating` and `.dangerous` tools are filtered out of
  `availableSchemas(...)` — the model literally doesn't see them.
- `.readOnly` and `.planning` tools run normally.
- `.network` tools still need consent; useful for letting the agent
  fetch a referenced URL while restricted from changing anything
  locally.

## Switching mode

Three entry points:

1. **`AgentEditSheet`** — Settings → Agents → edit → "Agent mode"
   segmented control. Persists into `agents.json`.
2. **`ModePickerView`** — pinned in the chat toolbar next to the
   model picker; flips the mode for the current conversation only
   (does not edit the saved AgentConfig).
3. **`/mode plan`** / **`/mode build`** — slash command interpreted
   by `SlashCommandLibrary`; same effect as the toolbar picker.

## Wiring into the chat flow

The mode reaches `InferenceService` through the agent record and
flows into the tool dispatcher via `ToolContext.mode`. Two practical
consequences:

1. The system block listing tools to the model is built from
   `registry.availableSchemas(mode: ctx.mode)`. Plan-mode agents
   therefore don't even *learn* about `write` / `edit` / `shell`,
   so they won't waste tokens proposing them.

2. The registry rejects late-arriving calls with a `denied` error
   if the mode changes mid-stream. The model sees a structured
   message it can recover from.

## Permission model summary

```
                  ┌───────────────────────────────┐
                  │  Tool dispatch (name, input)  │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
            ┌─────────────────────────────────────────┐
            │  Mode filter:                           │
            │    plan + (mutating | dangerous) → DENY │
            └─────────────────────────────────────────┘
                                  │
                                  ▼
            ┌─────────────────────────────────────────┐
            │  PermissionStore default (durable)      │
            │    alwaysAllow → run                    │
            │    alwaysDeny  → return 'denied'        │
            │    ask         → ↓                      │
            └─────────────────────────────────────────┘
                                  │
                                  ▼
            ┌─────────────────────────────────────────┐
            │  Session cache (ToolRegistry)           │
            │    cached → run                         │
            │    miss   → ↓                           │
            └─────────────────────────────────────────┘
                                  │
                                  ▼
            ┌─────────────────────────────────────────┐
            │  PermissionPromptView (modal)           │
            │    Deny | Allow once | Always allow     │
            └─────────────────────────────────────────┘
```

## Out of scope (today)

- Per-tool override at the *agent* level (the `allowedToolNames` set
  already does it for inclusion; an "ask anyway even if allowed"
  override is not implemented).
- Cross-session reset of "always allow" cache; today the cache lives
  for the registry's lifetime. The Permissions tab has a reset button
  for session grants; the durable layer is on `PermissionStore`.
- Permission audit log — every grant is in memory only.
