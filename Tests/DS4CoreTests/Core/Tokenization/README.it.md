# Test di tokenizzazione

[`Backends/DeepSeekV4/`](Backends/DeepSeekV4/README.md) copre il caricamento
del vocabolario, il fallback a livello di byte, i token speciali, i
round-trip encode/decode e la compatibilità della storica API del tokenizer.

[`Backends/GLM52/`](Backends/GLM52/README.md) copre il pretokenizer `glm4`,
il BPE, gli ID reali dei token speciali, la politica di stop, i round-trip a
livello di byte e la factory per architettura.

Aggiungi fixture per Unicode, sequenze di byte non valide, token speciali
adiacenti e input vuoto quando il comportamento del tokenizer cambia. Evita
asserzioni legate a un percorso di modello locale.
