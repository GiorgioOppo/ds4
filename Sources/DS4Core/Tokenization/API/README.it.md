[English](README.md) | **Italiano**

# Tokenization/API

Contratto minimo condiviso dai tokenizer dei diversi backend.

`TokenizerProtocol` espone tokenizzazione del testo, tokenizzazione di prompt già
resi e decodifica di un token. Non impone id per ruoli, reasoning o tool: quei
delimitatori appartengono al backend conversazionale e possono non essere token
atomici in tutte le famiglie.

`TokenizerFactory` seleziona esclusivamente dall'architettura rilevata:

- `deepseek4` → `DeepSeekV4Tokenizer`;
- `glm-dsa` → `GLM52Tokenizer`;
- Qwen riconosciuto → errore esplicito `tokenizerNotImplemented`;
- architetture sconosciute o assenti → errore esplicito.

Il fallback DeepSeek storico si applica soltanto quando
`general.architecture` manca e sono presenti metadati `deepseek4.*`. Non viene
mai applicato a un'architettura esplicita diversa.

La disponibilità del tokenizer è separata dalla disponibilità del runtime:
GLM 5.2 può essere ispezionato e tokenizzato anche mentre il suo backend Metal
rimane `recognizedButNotImplemented`. `ConversationBackendPolicy` applica la
stessa separazione scegliendo `deepSeekDSML` o `glm52Native`, senza costruire un
motore di inferenza.
