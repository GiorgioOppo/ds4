# DwarfStar/Chat

La chat: view-model + UI.

- **`ChatStore.swift`** — view-model `@MainActor @Observable`. Possiede l'`InferenceService`, fa da mirror del suo stream di eventi, gestisce il loop dei tool (incluso l'instradamento di `subagent_run` all'engine), gli allegati di testo, l'avviso di contesto quasi pieno, e i settaggi (cache esperti, KV su disco, raw-KV ring).
- **`ChatView.swift`** — trascrizione + composer + renderer Markdown + le sottoview dei messaggi (reasoning, tool-call/result, sub-agent, chip allegati).
- **`ChatTabView.swift`** — wrapper del tab (header, menu progetto/agente/tool).
- **`ContentView.swift`** — schermata di caricamento/onboarding del modello con la sezione settings.

Candidati a split (review): estrarre `MarkdownView` e le `MessageViews` da `ChatView.swift`.
