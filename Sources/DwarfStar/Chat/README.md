# DwarfStar/Chat

Chat view model and UI.

- **`ChatStore.swift`** is the `@MainActor @Observable` view model. It owns
  `InferenceService`, mirrors its event stream, manages the tool loop including
  routing `subagent_run` back into the engine (malformed calls — bad JSON,
  missing question, unknown agent id, ungrantable tools — are rejected with an
  explanatory error fed back to the model and shown in the transcript; valid
  runs appear immediately as an in-progress card that streams the sub-agent's
  internal steps live and keeps them on error/stop), handles text attachments, shows
  near-context-full warnings, applies memory settings such as expert cache, disk
  KV, and raw-KV ring, and manages **multiple persistent chats**. The active chat
  lives in `messages`; inactive chats are stored on disk.
- **`ChatSession.swift`** defines the `Codable` chat model, including metadata and
  transcript entries as `StoredMessage`. `ChatSessionStore` persists one JSON
  file per chat under `Application Support/DwarfStar/chats`.
- **`ChatView.swift`** renders the transcript, composer, Markdown, reasoning
  sections, tool calls/results, sub-agent events, and attachment chips.
- **`ChatListView.swift`** is the saved-chat popover for switching, renaming,
  deleting, and creating chats.
- **`ChatTabView.swift`** wraps the tab header and project/agent/tool menus.
- **`ContentView.swift`** handles model loading/onboarding and the embedded
  settings section.

## Reopening A Chat

After app restart or after switching chats, the engine no longer owns the KV for
that conversation. On the first new send, the visible history is rendered again
through `InferenceService.sendWithHistory`. Disk KV can restore the prefix, then
subsequent turns go back to incremental append-only execution.

## Future Split Candidates

`ChatView.swift` still contains a lot of rendering code. `MarkdownView` and the
message subviews are good candidates for extraction during a focused UI cleanup.
