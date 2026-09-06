**English** | [Italiano](README.it.md)

# Chat Views

SwiftUI presentation for conversations:

- `ChatTabView` chooses local, distributed, loading, or onboarding content.
- `ChatView` composes the transcript, toolbar, and prompt editor.
- `ChatEmptyState` offers editable starter prompts and contains the focused
  `ChatResponseSettings` popover for sampling and reasoning controls.
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
The toolbar uses two rows so model identity and conversation actions stay
visible at the minimum window width. Conversation history supports title search,
visible row actions, and confirmation before deletion. The composer supports
Command-Return to send and Escape to stop generation; Command-N starts a new
conversation. Image previews and a Vision indicator are gated by the configured
image capability, while text attachments remain available.

During token streaming, transcript autoscroll is non-animated and capped at
five updates per second. This prevents overlapping `ScrollViewProxy`
animations from surviving the teardown of the Chat panel when the user moves
to another sidebar section; generation remains owned by `ChatStore` and keeps
running across navigation.
Scrolling toward older messages pauses following; a button returns to the most
recent message and resumes following. Reaching the bottom manually also resumes
it. The content geometry distinguishes user scrolling from an expanding response.
