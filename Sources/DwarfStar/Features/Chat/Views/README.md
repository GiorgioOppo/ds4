# Chat Views

SwiftUI presentation for conversations:

- `ChatTabView` chooses local, distributed, loading, or onboarding content.
- `ChatView` composes the transcript, toolbar, and prompt editor.
- `MessageRow`, `MarkdownView`, and `ToolMessageViews` render message content.
- `AttachmentViews`, `ToolSheets`, and `ChatListView` provide focused controls.
- `ContentView` contains an older model-load flow and is not the current root.

Views observe `ChatStore` and should remain declarative. Put session,
generation, tool, and model-lifecycle decisions in the view model. If a new
visual component becomes independently reusable or substantial, give it a
focused file instead of extending `ChatView.swift`.

