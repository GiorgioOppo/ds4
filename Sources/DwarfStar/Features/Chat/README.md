# DwarfStar/Features/Chat

Chat view model and UI.

- **`ViewModels/ChatStore.swift`** is the `@MainActor @Observable` view model. It owns
  `InferenceService`, mirrors its event stream, manages the tool loop including
  routing `subagent_run` back into the engine (malformed calls — bad JSON,
  missing question, unknown agent id, ungrantable tools — are rejected with an
  explanatory error fed back to the model and shown in the transcript; valid
  runs appear immediately as an in-progress card that streams the sub-agent's
  internal steps live and keeps them on error/stop), handles text attachments, shows
  near-context-full warnings, applies memory settings such as expert cache,
  expert pread, expert bundle, dense streaming, `mlock`, Q4 dense cache, disk KV,
  and raw-KV ring, and manages **multiple persistent chats**. The active chat
  lives in `messages`; inactive chats are stored on disk. It also owns the
  sampling and prefill knobs (`DS4Temperature`, `DS4RepPenalty`,
  `DS4PrefillUnion`, `DS4PrefillChunk`); all keys and defaults are listed in
  the root [Configuration Reference](../../../../README.md#configuration-reference).
- **`Persistence/ChatSession.swift`** defines the `Codable` chat model, including metadata and
  transcript entries as `StoredMessage`. `ChatSessionStore` persists one JSON
  file per chat under `Application Support/DwarfStar/chats`.
- **`Views/ChatView.swift`** owns the transcript and composer layout. Focused
  view files render Markdown, messages, tools, sheets and attachments.
- **`Views/ChatListView.swift`** is the saved-chat popover for switching, renaming,
  deleting, and creating chats.
- **`Views/ChatTabView.swift`** switches the Chat tab by engine mode and phase: the
  local `ChatView` when ready, a loading/onboarding placeholder otherwise, and
  `CoordinatorChatView` in Distributed mode. The header with the
  project/agent/tool menus lives in `ChatView`.
- **`Views/ContentView.swift`** contains `ModelLoadView`, the pre-load configuration
  form (model selection, memory settings, context, agent, system prompt).
  Neither view is referenced by the current UI — `RootView`/`ChatTabView` and
  the Settings tab took over this flow, so the file is a removal candidate.

## Reopening A Chat

After app restart or after switching chats, the engine no longer owns the KV for
that conversation. On the first new send, the visible history is rendered again
through `InferenceService.sendWithHistory`. Disk KV can restore the prefix, then
subsequent turns go back to incremental append-only execution.

## Shared Engine

When the model is loaded, `ChatStore.sharedEngine` exposes the one local
`InferenceService` used by Chat, Server, and local Benchmark. Server start fails
until this engine is ready. Benchmark is allowed only while chat is idle because
benchmark runs rewrite KV state.

## File Ownership

Conversation DTOs are in `Models`, storage conversion and session files in
`Persistence`, mutable UI state in `ViewModels`, and all rendering in `Views`.
Shared engine types stay in `DS4Engine` or `DS4Core` rather than moving into the
GUI feature.

See [`FLOW.md`](FLOW.md) for the end-to-end message, tool, session, and
shared-engine lifecycle. Each child directory has its own ownership and change
rules in a local README.
