[English](README.md) | **Italiano**

# Protocollo conversazionale Laguna S 2.1

Questa directory possiede il framing chat e la sintassi nativa dei tool di
Laguna S 2.1. Il formato inizia con il marcatore di sequenza `〈|EOS|〉`,
inquadra i ruoli con tag testuali in stile XML (`<system>`, `<user>`,
`<tool_response>`) e riserva token di controllo dedicati solo a `<assistant>`,
`</assistant>`, `<think>`, `</think>`, `<tool_call>` e `</tool_call>`. Le
chiamate ai tool usano la grammatica piatta
`<tool_call>nome<arg_key>…</arg_key><arg_value>…</arg_value>` che l'upstream
condivide con GLM, dichiarata in una sezione di sistema `### Tools` racchiusa
da `<available_tools>`. I turni assistant portano il reasoning interlacciato
tra `<think>` e `</think>` e chiudono con `</assistant>`, il token di fine
turno della famiglia; il contenuto di reasoning va preservato tra le chiamate
ai tool.

Renderer e parser sono indipendenti dal modello e testabili senza GGUF. Il
testo non fidato di system, user, argomenti e risultati dei tool viene
neutralizzato prima che il transcript renderizzato venga scandito per i token
speciali; i tag di ruolo testuali fanno parte dell'insieme neutralizzato
perché guidano il modello pur non essendo token di vocabolario.
`LagunaConversationProtocol.SamplingDefaults` registra i default di sampling
di riferimento (temperatura 0.7, top-k 20, top-p 0.95, min-p 0.05). La
registrazione del runtime resta fuori da questa directory; avere tokenizer e
renderer non implica che esista un grafo di inferenza Metal.
