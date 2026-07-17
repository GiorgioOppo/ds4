# Chat Persistence

This directory owns the on-disk representation of app conversations.

- `ChatSession.swift` defines the `Codable` session, message, tool-call, and
  sub-agent records and the JSON-backed `ChatSessionStore`.
- `ChatMessageMapping.swift` converts between persisted records, UI messages,
  and engine roles.
- `DS4Engine/Inference/Autotuning/MachineAutoTuneTransactionStore.swift` owns
  the crash-safe two-phase commit for a validated machine-auto-tune winner. It
  records `prepared → installed → committing → committed` atomically and
  restores the complete initial knob snapshot at the next launch if any
  non-terminal phase was interrupted.

The flow is `UIMessage` -> `StoredMessage` -> one JSON file per chat under
Application Support, and the reverse when a session is opened. Preserve
backward-compatible decoding when adding fields; use defaults or optional
properties rather than making existing chat files unreadable.
