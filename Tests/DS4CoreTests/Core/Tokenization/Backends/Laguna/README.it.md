[English](README.md) | **Italiano**

# Test del tokenizer Laguna

Fixture deterministiche (token di controllo più i 256 singoli byte-level)
coprono il pre-split sui newline di Laguna e la sua differenza osservabile su
CRLF rispetto allo splitter GLM4 puro, i gruppi numerici a una cifra, i merge
BPE, i round-trip byte-level, i controlli atomici del transcript renderizzato
contro i tag di ruolo testuali, la politica di stop EOS/EOT e la
neutralizzazione dei token di controllo — tutto senza scaricare un GGUF.
