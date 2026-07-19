[English](README.md) | **Italiano**

# Registrazione runtime GLM 5.2

Questa directory possiede la registrazione lato motore per i modelli GGUF che
dichiarano `general.architecture = "glm-dsa"`.

`GLM52BackendDefinition` attualmente pubblicizza solo capability portabili del
modello. Il suo insieme di capability runtime resta intenzionalmente vuoto, e
la selezione del backend restituisce `backendNotImplemented`, finché il grafo
Metal residente, il tokenizer e le fixture di qualità non passano insieme.
Questo confine impedisce che un GGUF GLM venga interpretato o eseguito come
DeepSeek V4.

L'implementazione numerica appartiene a un albero gemello
`Sources/DS4Metal/Backends/GLM52`; i metadati di architettura e gli schemi dei
tensori appartengono a DS4Core. Non aggiungere condizionali GLM all'hot loop
di DeepSeek.
