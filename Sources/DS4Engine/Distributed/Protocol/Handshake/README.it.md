# Protocol/Handshake

Definisce i messaggi che trasformano un worker inattivo in un nodo assegnato.

## Tipi

- `DistHello`: versione, stato corrente, modello e slice già caricata.
- `DistAssign`: modello, contesto, slice, output head, budget KV, cache esperti,
  sidecar/Q4, usage profile e knob prestazionali whitelisted.

## Flusso e dipendenze

Il worker invia `HELLO`; il coordinator verifica compatibilità e, dopo i file,
invia `ASSIGN`. La risposta `READY` riusa il payload `DistHello`. I file sono
negoziati dai tipi in [`Files`](../Files/README.md).

## Estensione

Ogni campo deve avere default o incompatibilità esplicita. Le variabili ambiente
ricevute devono essere filtrate anche dal worker usando la whitelist in
[`Core`](../Core/README.md); non trasportare segreti o impostazioni arbitrarie.
