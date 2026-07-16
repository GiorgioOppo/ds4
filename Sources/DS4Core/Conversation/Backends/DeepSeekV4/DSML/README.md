# Conversation/Backends/DeepSeekV4/DSML

Implementazione del protocollo DSML usato dal modello per dichiarare ed emettere
chiamate a strumenti.

## File principali

- [`ToolMarkup.swift`](ToolMarkup.swift): costruisce delimitatori e tag DSML.
- [`ChatRenderer.swift`](ChatRenderer.swift): rende cronologia, schemi tool e
  prompt di generazione, con modalità completa o compatta.
- [`ToolCallParser.swift`](ToolCallParser.swift): estrae invocazioni e parametri
  dall'output del modello.

## Flusso

I tipi in [`Models`](../../../Models/README.md) entrano nel renderer; il testo prodotto
viene tokenizzato. Dopo la generazione, il parser ricostruisce `[ToolCall]`; il
risultato dello strumento torna nella cronologia come `ChatTurn.toolResult`.

## Regole di modifica

- Preservare delimitatori, escaping di `</tool_result>` e ordinamento stabile JSON.
- Trattare input del modello e risultati tool come dati non fidati.
- Misurare la modalità compatta sia per riduzione del prefill sia per affidabilità.
- Aggiungere test round-trip quando cambiano rendering o parsing.
