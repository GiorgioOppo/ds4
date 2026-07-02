# DwarfStar/Chat

La chat: view-model + UI.

- **`ChatStore.swift`** — view-model `@MainActor @Observable`. Possiede l'`InferenceService`, fa da mirror del suo stream di eventi, gestisce il loop dei tool (incluso l'instradamento di `subagent_run` all'engine), gli allegati di testo, l'avviso di contesto quasi pieno, i settaggi (cache esperti, KV su disco, raw-KV ring) e le **chat persistenti multiple** (creare/cambiare/rinominare/eliminare; la chat attiva è in `messages`, le altre su disco).
- **`ChatSession.swift`** — modello `Codable` di una chat (metadati + trascrizione come `StoredMessage`) e `ChatSessionStore`: persistenza su disco (un JSON per chat in `Application Support/DwarfStar/chats`).
- **`ChatView.swift`** — trascrizione + composer + renderer Markdown + le sottoview dei messaggi (reasoning, tool-call/result, sub-agent, chip allegati).
- **`ChatListView.swift`** — popover con l'elenco delle chat salvate (cambia, rinomina, elimina, nuova).
- **`ChatTabView.swift`** — wrapper del tab (header, menu progetto/agente/tool).
- **`ContentView.swift`** — schermata di caricamento/onboarding del modello con la sezione settings.

**Riapertura di una chat** (dopo la chiusura dell'app o cambiando chat): l'engine
non possiede più la KV di quella conversazione, quindi al primo invio la storia
visibile viene ri-renderizzata (`InferenceService.sendWithHistory`) — la cache KV
su disco ripristina il prefisso, poi i turni tornano incrementali.

Candidati a split (review): estrarre `MarkdownView` e le `MessageViews` da `ChatView.swift`.
