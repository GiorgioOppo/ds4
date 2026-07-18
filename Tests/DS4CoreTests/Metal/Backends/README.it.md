# Backend Metal Tests

Test organizzati per famiglia di modello. Le suite comuni di runtime e kernel
restano nelle cartelle di primo livello; qui vivono shape, schema GGUF, stato KV
e composizioni di grafo che hanno semantica specifica del backend.

- [`DeepSeekV4/`](DeepSeekV4/README.md): backend attualmente implementato.
- [`GLM52/`](GLM52/README.md): schema/DSA, oracoli e primitive Metal progressive;
  nessun test dichiara ancora disponibile il decoder.
