# Conversation

Contratti portabili per costruire una conversazione e convertirla nel prompt
atteso dal modello DeepSeek-V4.

## Contenuto

- [`Models/`](Models/README.md): `ToolSpec`, `ToolCall` e `ChatTurn`.
- [`DSML/`](DSML/README.md): markup, rendering e parsing delle chiamate tool.

## Flusso e dipendenze

Il livello applicativo crea una sequenza di `ChatTurn`; `ChatRenderer` la rende
nel template chat usando i token definiti da `ToolMarkup`; il risultato passa a
[`Tokenizer`](../Tokenization/README.md). L'output del modello viene esaminato da
`ToolCallParser` per ricostruire eventuali chiamate. La cartella dipende solo dai
tipi di `DS4Core` e da Foundation.

## Regole di modifica

Il testo del template e i delimitatori sono parte del protocollo addestrato:
ogni variazione deve essere confrontata con `tokenizer.chat_template`, coperta da
test e valutata anche per il costo in token di prefill.
