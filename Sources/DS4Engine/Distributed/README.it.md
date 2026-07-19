[English](README.md) | **Italiano**

# Distributed

Implementa inferenza su più Mac con due strategie:

- **pipeline orizzontale**: ogni worker possiede un intervallo contiguo di layer;
- **parallelismo verticale degli esperti**: il coordinator esegue il backbone e
  aggrega contributi MoE prodotti da shard remoti.

Il protocollo wire corrente è `Dist.protocolVersion = 11` e richiede uguaglianza
stretta fra i nodi.

La geometria viene letta dal GGUF: la pipeline copre 43 layer/256 esperti per
Flash e 61 layer/384 esperti per Pro. Il runtime distribuito accetta il GGUF Pro
Q2 completo; i due file del package Pro Q4 non sono slice eseguibili e restano
solo scaricabili finché non esisterà un loader multi-shard.

## Struttura

- [`Protocol`](Protocol/README.it.md): framing e messaggi serializzabili.
- [`Transport`](Transport/README.it.md): connessioni TCP basate su Network.framework.
- [`Coordinator`](Coordinator/README.it.md): topologia, chat, KV e distribuzione file.
- [`Worker`](Worker/README.it.md): listener, assegnazioni ed esecuzione richieste.
- [`Execution`](Execution/README.it.md): adattatori del decoder per slice e shard.
- [`Files`](Files/README.it.md): hash, manifest e archivio locale dei modelli.

Il formato e le sequenze wire sono descritti in
[`PROTOCOLLO.md`](PROTOCOLLO.it.md).

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
