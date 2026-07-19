[English](README.md) | **Italiano**

# Test dei formati

Test per i formati persistenti e dei file modello di proprietà di `DS4Core`:

- `GGUF/`: metadati del container e descrittori dei tensori.
- `KVCheckpoint/`: codifica e validazione dei checkpoint KV.
- `Quantization/`: conversione numerica e helper di quantizzazione.

I test dei formati devono includere input malformati/troncati e preservare la
compatibilità con i dati già scritti ogniqualvolta il formato è stabile.
