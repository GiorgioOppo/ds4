# Chat Views

SwiftUI presentation for conversations:

- `ChatTabView` chooses local, distributed, loading, or onboarding content.
- `ChatView` composes the transcript, toolbar, and prompt editor.
- `MessageRow`, `MarkdownView`, and `ToolMessageViews` render message content.
- `AttachmentViews`, `ToolSheets`, and `ChatListView` provide focused controls.
- `ContentView` contains an older model-load flow and is not the current root.

The legacy pre-load form still opens the same `DownloadView` used by Settings;
it must not define its own remote model list. Current model acquisition should
be documented and evolved under `Features/ModelManagement`.

Views observe `ChatStore` and should remain declarative. Put session,
generation, tool, and model-lifecycle decisions in the view model. If a new
visual component becomes independently reusable or substantial, give it a
focused file instead of extending `ChatView.swift`.

The header uses the inspected/loaded model descriptor and never assumes a
DeepSeek fallback. Tool and reasoning controls are rendered only when the
selected backend advertises the corresponding runtime capability.

During token streaming, transcript autoscroll is non-animated and capped at
five updates per second. This prevents overlapping `ScrollViewProxy`
animations from surviving the teardown of the Chat panel when the user moves
to another sidebar section; generation remains owned by `ChatStore` and keeps
running across navigation.
