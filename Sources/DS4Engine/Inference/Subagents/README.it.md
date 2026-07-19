[English](README.md) | **Italiano**

# Inference/Subagents

Esegue attività delegate in un contesto isolato, preservando la conversazione
principale.

## Componente e flusso

`InferenceService+Subagents.swift` risolve ruolo, target e tool concessi;
prepara un prefisso dedicato; salva la KV principale; esegue round limitati di
inferenza/tool; infine ripristina il contesto originale. I prefissi di file o
progetto possono essere riusati tramite una cache KV content-addressed.
Contenuto di progetto, domanda, prompt di ruolo, schemi e risultati tool vengono
neutralizzati prima di aggiungere il framing fidato della chat isolata.

## Dipendenze

Usa [`Agents`](../../Agents/README.it.md), [`Projects`](../../Projects/README.it.md),
[`Tools`](../../Tools/README.it.md) e [`Persistence/KV`](../../Persistence/KV/README.it.md).

## Estensione

- Intersecare sempre i tool richiesti sia con `subAgentGrantable` sia con
  `allowedTools`, lo scope fidato catturato dal profilo padre. Il ruolo e gli
  argomenti scelti dal modello possono restringere lo scope, mai ampliarlo.
- Imporre limiti a round, token e testo riportato nel trace.
- Ripristinare la KV principale anche in caso di errore o cancellazione.
- Non condividere implicitamente contenuti o autorizzazioni fra agenti.
