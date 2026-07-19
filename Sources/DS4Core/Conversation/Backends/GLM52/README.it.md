[English](README.md) | **Italiano**

# Protocollo di conversazione GLM 5.2

Questa directory possiede il framing della chat GLM 5.2 e la sintassi nativa
dei tool. Il formato wire inizia con `[gMASK]<sop>`, usa token di controllo
`<|role|>` e codifica le chiamate ai tool come
`<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value>`.

Il renderer e il parser sono indipendenti dal modello e testabili senza un
GGUF. Il testo non fidato di system, utente, argomenti e risultati dei tool
viene neutralizzato prima che una trascrizione renderizzata venga scansionata
alla ricerca di token speciali. La registrazione a runtime resta fuori da
questa directory; avere un tokenizer e un renderer non implica che il grafo di
inferenza Metal sia completo.

`GLM52ChatRenderer` segue il loop dei messaggi del vero
`tokenizer.chat_template`: i messaggi system restano nelle loro posizioni
originali nella trascrizione invece di essere raccolti in un prologo. Una
sequenza di risultati di tool consecutivi apre `<|observation|>` una sola
volta e poi emette un wrapper `<tool_response>…</tool_response>` per ciascun
risultato; qualsiasi ruolo intermedio avvia una nuova sequenza di
observation.
