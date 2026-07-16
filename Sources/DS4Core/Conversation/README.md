# Conversation

Tipi portabili per la conversazione e formati di rendering specifici dei
backend.

## Contenuto

- [`Models/`](Models/README.md): `ToolSpec`, `ToolCall` e `ChatTurn`.
- [`Backends/DeepSeekV4/`](Backends/DeepSeekV4/README.md): template e protocollo
  tool DSML di DeepSeek V4.
- [`Backends/Qwen/`](Backends/Qwen/README.md): punto di estensione documentato,
  senza renderer o parser fittizi.

## Flusso e dipendenze

Il livello applicativo crea una sequenza di `ChatTurn`; il backend DeepSeek usa
`ChatRenderer` e `ToolMarkup` per produrre il proprio template, che passa a
[`Tokenizer`](../Tokenization/README.md). L'output del modello viene esaminato da
`ToolCallParser` per ricostruire eventuali chiamate. La cartella dipende solo dai
tipi di `DS4Core` e da Foundation.

## Regole di modifica

Template e delimitatori sono parte del protocollo addestrato del singolo backend:
ogni variazione deve essere confrontata con `tokenizer.chat_template`, coperta da
test e valutata anche per il costo in token di prefill.
