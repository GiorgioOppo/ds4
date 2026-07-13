# Distributed/Coordinator

`DistCoordinator` possiede la configurazione del cluster, il motore locale e le
connessioni ai worker. Le responsabilità sono separate in estensioni.

## File

- `DistCoordinator.swift`: stato, `Peer`, `Config` e route attiva.
- `+Connections`: partizionamento, handshake, transfer e assegnazione.
- `+Files`: costruzione dell'offerta e streaming dei file.
- `+KV`: negoziazione e salvataggio dei checkpoint shard.
- `+HorizontalChat`: pipeline per slice di layer.
- `+VerticalChat` e `+ExpertParallelism`: backbone locale e shard esperti.
- `+Benchmark`: misure per entrambe le topologie.

## Dipendenze e flusso

Usa [`Protocol`](../Protocol/README.md), [`Transport`](../Transport/README.md),
[`Execution`](../Execution/README.md) e [`Files`](../Files/README.md). `connect`
prepara la route orizzontale; `connectVertical` prepara gli shard esperti. Solo
dopo tutti i `READY` una chat o un benchmark può iniziare.

## Estensione

Mantenere la configurazione immutabile durante un turno, associare ogni
risultato alla sessione corrente e chiudere connessioni/return listener in ogni
percorso di errore. Scheduling e retry restano qui, non nei tipi del protocollo.
