# DS4Core

Core dell'engine, **puro Swift senza Metal**: parsing del modello, tokenizer, campionamento e formato della chat/tool. È la base condivisa da `DS4Metal` e `DS4Engine`, ed è la più coperta dai test (`Tests/DS4CoreTests`).

- **`Format/`** — formati su disco: GGUF (mmap), Half (f16), KVCFile (checkpoint KV su disco).
- **`Inference/`** — shape del modello, tokenizer BPE (token di controllo), sampler, rendering chat + parser DSML dei tool.
- **`Streaming/`** — pianificazione della cache SSD e lock di memoria simulato.

Nessuna dipendenza esterna; nessun link a Metal (così gira/compila ovunque e i test sono veloci).
