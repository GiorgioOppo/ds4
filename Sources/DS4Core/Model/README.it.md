[English](README.md) | **Italiano**

# Model

Rilevamento portabile dell'architettura e configurazioni isolate per backend.

## Struttura

- [`Common/`](Common/README.it.md): identificatore canonico, famiglia, capability,
  descriptor e rilevamento da `general.architecture`.
- [`Backends/DeepSeekV4/`](Backends/DeepSeekV4/README.it.md): forma e validazione
  dei metadata DeepSeek V4, con alias compatibili delle API storiche.
- [`Backends/GLM52/`](Backends/GLM52/README.it.md): geometria GLM 5.2 e
  validazione stretta del namespace `glm-dsa`.
- [`Backends/Qwen/`](Backends/Qwen/README.it.md): punto di estensione documentato;
  il backend Qwen non è ancora implementato.

## Flusso e dipendenze

Il detector classifica prima l'architettura; solo dopo il backend corrispondente
può leggere e validare il proprio namespace metadata. Le costanti specifiche dei
kernel DeepSeek-V4 restano in
[`DS4Metal/Backends/DeepSeekV4/Architecture`](../../DS4Metal/Backends/DeepSeekV4/Architecture/README.it.md).

## Regole di modifica

Non interpretare un'architettura riconosciuta ma non implementata tramite il
backend DeepSeek. Nuove varianti devono avere validazione esplicita e valori
derivati controllati; non duplicare qui costanti già lette dal GGUF.
