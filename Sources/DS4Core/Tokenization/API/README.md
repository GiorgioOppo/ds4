# Tokenization/API

Contratto minimo condiviso dai tokenizer dei diversi backend.

`TokenizerProtocol` espone tokenizzazione del testo, tokenizzazione di prompt già
resi e decodifica di un token. Non impone id per ruoli, reasoning o tool: quei
delimitatori appartengono al backend conversazionale e possono non essere token
atomici in tutte le famiglie.

