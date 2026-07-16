# Runtime/Common

- `ModelInspector` converte i metadati GGUF nei descrittori neutrali di DS4Core e
  nelle capability effettivamente implementate da DS4Engine.
- `BackendSelector` distingue backend disponibile, famiglia riconosciuta ma non
  implementata e architettura sconosciuta.
- `RuntimeBackendFactory` è il confine da chiamare prima di `ModelConfig`,
  tokenizer o allocazioni specifiche dell'architettura.
- `BackendCapabilities` guida GUI, diagnostica e servizi opzionali. Non va usato
  per descrivere proprietà teoriche non ancora supportate dal runtime.

I tipi di questa cartella non eseguono inferenza e non sono nel percorso per-token.
