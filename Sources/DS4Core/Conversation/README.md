# Conversation

Tipi portabili per la conversazione e formati di rendering specifici dei
backend.

## Contenuto

- [`Models/`](Models/README.md): `ToolSpec`, `ToolCall` e `ChatTurn`.
- [`Backends/DeepSeekV4/`](Backends/DeepSeekV4/README.md): template e protocollo
  tool DSML di DeepSeek V4.
- [`Backends/GLM52/`](Backends/GLM52/README.md): ruoli GLM, reasoning,
  tool-call XML piatto e parser incrementale contenuto.
- [`Backends/Qwen/`](Backends/Qwen/README.md): punto di estensione documentato,
  senza renderer o parser fittizi.

## Flusso e dipendenze

Il livello applicativo crea una sequenza di `ChatTurn`; la policy frontend
seleziona DSML per DeepSeek oppure il protocollo nativo GLM. Il testo reso passa
al tokenizer della stessa architettura e il parser corrispondente ricostruisce
eventuali chiamate. La cartella dipende solo dai tipi di `DS4Core` e da
Foundation.

## Regole di modifica

Template e delimitatori sono parte del protocollo addestrato del singolo backend:
ogni variazione deve essere confrontata con `tokenizer.chat_template`, coperta da
test e valutata anche per il costo in token di prefill.
