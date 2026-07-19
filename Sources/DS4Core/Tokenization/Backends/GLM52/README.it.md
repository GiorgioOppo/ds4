[English](README.md) | **Italiano**

# Tokenizer GLM 5.2

`GLM52Tokenizer` implementa il BPE byte-level in stile GPT-2 con il
pre-tokenizer `glm4` di ChatGLM4, controlli nativi atomici di ruolo/strumento,
la detokenizzazione, token di stop consapevoli del reasoning e la codifica dei
prompt `[gMASK]<sop>`.

L'inizializzatore da modello convalida `general.architecture`, le tabelle del
tokenizer, il tipo di pre-tokenizer e tutti i token di protocollo richiesti.
Un inizializzatore interno senza modello e lo splitter puro `GLM4Pretokenizer`
supportano unit test deterministici senza scaricare o mappare un GGUF GLM.

Questo livello non seleziona un backend di inferenza. Il rilevamento del
modello può riconoscere GLM mentre il grafo Metal resta esplicitamente non
disponibile.
