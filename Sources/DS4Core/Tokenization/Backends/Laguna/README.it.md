[English](README.md) | **Italiano**

# Tokenizer Laguna S 2.1

`LagunaTokenizer` implementa il BPE byte-level GPT-2 con il pre-tokenizer
Laguna: le sequenze di byte LF vengono prima separate dagli span senza
newline, poi ogni span passa attraverso la forma di split in stile GLM4 con
gruppi numerici a una sola cifra. Il pre-split sui newline è osservabile per
CRLF — il CR resta nello span precedente e LF ne apre uno nuovo, quindi non
possono fondersi in un unico pezzo BPE.

L'inizializzatore da modello valida `general.architecture`, le tabelle del
tokenizer e i token di controllo richiesti (`<assistant>`, `</assistant>`,
`<think>`, `</think>`, `<tool_call>`, `</tool_call>`); gli id BOS/EOS sono
metadati obbligatori e l'id di fine turno ricade sulla voce di vocabolario
`</assistant>`. L'upstream seleziona il pre-tokenizer in base alla famiglia
del modello, non da `tokenizer.ggml.pre`, quindi quella chiave qui non viene
deliberatamente vincolata. Un inizializzatore interno senza modello e lo
splitter puro `LagunaPretokenizer` supportano test deterministici senza
scaricare o mappare un GGUF Laguna.

Questo strato non seleziona un backend di inferenza. Il rilevamento del
modello può riconoscere Laguna mentre il grafo Metal resta esplicitamente non
disponibile.
