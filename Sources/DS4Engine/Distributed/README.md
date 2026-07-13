# Distributed

Implementa inferenza su più Mac con due strategie:

- **pipeline orizzontale**: ogni worker possiede un intervallo contiguo di layer;
- **parallelismo verticale degli esperti**: il coordinator esegue il backbone e
  aggrega contributi MoE prodotti da shard remoti.

Il protocollo wire corrente è `Dist.protocolVersion = 10` e richiede uguaglianza
stretta fra i nodi.

## Struttura

- [`Protocol`](Protocol/README.md): framing e messaggi serializzabili.
- [`Transport`](Transport/README.md): connessioni TCP basate su Network.framework.
- [`Coordinator`](Coordinator/README.md): topologia, chat, KV e distribuzione file.
- [`Worker`](Worker/README.md): listener, assegnazioni ed esecuzione richieste.
- [`Execution`](Execution/README.md): adattatori del decoder per slice e shard.
- [`Files`](Files/README.md): hash, manifest e archivio locale dei modelli.

Il formato e le sequenze wire sono descritti in
[`PROTOCOLLO.md`](PROTOCOLLO.md).

## Flusso sintetico

1. Il coordinator si connette e verifica `HELLO` e versione.
2. Offre GGUF e sidecar; il worker richiede solo file mancanti o incompleti.
3. Il coordinator invia `ASSIGN`; il worker carica la responsabilità e risponde
   `READY`.
4. I `WORK` attraversano la route e producono `RESULT`, oppure i messaggi expert
   producono somme parziali.
5. A fine turno i nodi possono salvare checkpoint KV delle rispettive parti.

## Sicurezza e vincoli

Il traffico è TCP in chiaro e il listener non autentica i peer: usare soltanto
reti fidate. Lunghezze, route, slice, sessioni e payload devono essere validati
prima di allocare memoria o invocare Metal. Solo i knob `DS4_*` presenti nella
whitelist possono attraversare `ASSIGN`.

## Estensione

Una modifica incompatibile al wire richiede incremento di `protocolVersion`,
codec simmetrici e test con payload troncati/ostili. Il trasporto non deve
contenere logica di scheduling; i messaggi non devono dipendere da coordinator
o worker.
