[English](README.md) | **Italiano**

# Conversation/Models

Tipi valore condivisi tra interfaccia, motore e renderer della chat.

## File principali

- [`ConversationModels.swift`](ConversationModels.swift): definisce `ToolSpec`,
  schema di una funzione disponibile; `ToolCall`, invocazione con argomenti JSON;
  `ChatTurn`, sequenza tipizzata di messaggi system, user, assistant e tool result.

I tipi sono `Sendable` ed `Equatable`; `ToolSpec` e `ToolCall` sono anche
`Identifiable`, così possono attraversare in sicurezza UI e servizi concorrenti.

## Flusso e regole

Questi modelli non eseguono strumenti e non conoscono il backend. Sono consumati
da [`DSML`](../Backends/DeepSeekV4/DSML/README.it.md) e dai livelli superiori. Aggiungere nuovi casi a
`ChatTurn` solo aggiornando renderer, parser, persistenza e test di exhaustiveness;
non inserire qui stato runtime o dipendenze applicative.
