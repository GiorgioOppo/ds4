# Inference/Subagents

Esegue attività delegate in un contesto isolato, preservando la conversazione
principale.

## Componente e flusso

`InferenceService+Subagents.swift` risolve ruolo, target e tool concessi;
prepara un prefisso dedicato; salva la KV principale; esegue round limitati di
inferenza/tool; infine ripristina il contesto originale. I prefissi di file o
progetto possono essere riusati tramite una cache KV content-addressed.

## Dipendenze

Usa [`Agents`](../../Agents/README.md), [`Projects`](../../Projects/README.md),
[`Tools`](../../Tools/README.md) e [`Persistence/KV`](../../Persistence/KV/README.md).

## Estensione

- Intersecare sempre i tool richiesti con `subAgentGrantable`.
- Imporre limiti a round, token e testo riportato nel trace.
- Ripristinare la KV principale anche in caso di errore o cancellazione.
- Non condividere implicitamente contenuti o autorizzazioni fra agenti.
