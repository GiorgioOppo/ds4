# Expert parallelism — scissione verticale del modello

Stato al 13 luglio 2026: **fasi A-C implementate nel protocollo v10**. Sono
attivi payload, worker expert shard, backbone del coordinatore, chat verticale e
benchmark GUI. Restano da ampliare la validazione multi-Mac e le ottimizzazioni
di fase D.

Prerequisito operativo: RTT inferiore a circa 1 ms fra i nodi, tramite bridge
Thunderbolt o Ethernet diretta. Con circa 41 layer routed, un RTT di 7 ms
aggiunge quasi 300 ms per token prima del lavoro utile; su Wi-Fi questa
topologia è normalmente perdente.

## Perché una scissione verticale

La pipeline orizzontale assegna intervalli di layer ai worker. Gli SSD lavorano
in sequenza lungo il percorso del token: il tempo dominante tende alla somma
dei gather delle slice.

Nella scissione verticale i 256 esperti di ogni layer sono distribuiti fra i
worker. Quando il router sceglie sei esperti, ogni SSD legge in parallelo la
propria quota e restituisce una somma parziale. Sul costo dominante il tempo può
avvicinarsi al massimo dei worker anziché alla somma, ma si paga un round-trip
di rete per layer routed.

## Architettura implementata

- **Coordinatore/backbone locale**: embedding, route/attention di tutti i
  layer, KV, compressori NSA, FFN shared, riduzioni HC e output head.
- **Worker expert shard**: sottoinsieme degli esperti valido per tutti i layer;
  nessun KV o stato di conversazione; gather, gate/up/down e somma pesata.
- **Protocollo v10**: `expertAssign`, `expertWork`, `expertSum`.
- **GUI**: toggle Vertical split, connessione, chat e benchmark dedicato.

`StreamingDecoder` espone una callback `remoteExperts` usata al posto del ramo
FFN routed locale. Il coordinatore carica il backbone con cache esperti locale
disattivata; gli expert shard usano bundle, slot-cache e knob assegnati.

## Flusso per token e layer routed

1. Il backbone calcola route/attention e produce id e pesi dei sei esperti.
2. Il coordinator raggruppa gli id per mask proprietaria.
3. Invia in parallelo `expertWork` con layer, sequenza, id, pesi e attivazione.
4. Ogni worker valida che gli id siano posseduti, esegue la FFN e invia
   `expertSum`.
5. Il coordinator somma le parziali e prosegue con shared FFN/riduzione del
   layer.

Le attivazioni e le somme verticali sono trasportate a 32 o 16 bit. Il valore 8
bit disponibile per la pipeline orizzontale non viene usato dal percorso
verticale corrente.

## Assegnazione degli esperti

`DistExpertAssign.expertMask` contiene 256 bit: il bit `e` indica il possesso
dell'esperto `e`. La partizione corrente è **round-robin** (`e % workerCount`),
con copertura esatta e senza sovrapposizioni.

Il bilanciamento greedy basato sulla usage imatrix non è ancora attivo. È una
possibile fase D: dovrebbe distribuire il carico osservato, non soltanto il
numero di esperti, mantenendo copertura esatta e configurazione riproducibile.

## Costi di comunicazione

Con attivazione di 4096 elementi, ogni layer coinvolto invia e riceve circa:

- 32 KiB complessivi a F32, oltre a header/id/pesi;
- 16 KiB complessivi a F16, oltre a header/id/pesi.

Su 41 layer routed sono circa 1,3 MiB/token a F32 o 0,65 MiB/token a F16 per
worker coinvolto nel caso semplice. La latenza, più della banda, è il vincolo:
il round-trip si ripete lungo la sequenza dei layer.

Queste sono stime di ordine di grandezza. Il benchmark deve misurare traffico e
latenza reali con lo stesso GGUF e la stessa cache della baseline locale.

## Trasferimento dei file

Il setup riusa il trasferimento resumable del protocollo v8: GGUF e sidecar
sono offerti con SHA-256 e checkpoint concatenati. Al momento il worker può
ricevere il bundle completo e aprire soltanto i record necessari alla propria
mask.

Un trasferimento futuro di un bundle fisicamente shardato ridurrebbe spazio e
tempo di setup, ma richiede un nuovo tipo di manifest e una strategia di
validazione dedicata.

## Stato delle fasi

- **A — completata**: design, frame v10, encode/decode bound-checked e test
  round-trip.
- **B — completata**: `ExpertShard`, assegnazione worker, mask, bundle/cache e
  serving `expertWork` -> `expertSum`.
- **C — completata**: backbone locale, scatter/gather remoto, chat verticale,
  toggle e benchmark.
- **D — aperta**: bilanciamento dalla usage imatrix, overlap aggiuntivo,
  eventuali file shard e campagna A/B multi-Mac.

## Criteri di validazione

1. Misurare RTT prima di connettere la route verticale.
2. Verificare copertura e unicità delle mask.
3. Confrontare F32 verticale con il percorso locale a sei esperti.
4. Misurare separatamente F16 per qualità e rete.
5. Registrare prefill, decode a regime, byte/token, banda e cache hit-rate.
6. Confrontare verticale, pipeline e locale con modello, prompt e warm-up
   identici.

L'obiettivo progettuale resta almeno 1,5x rispetto al locale a parità di
qualità. Se la latenza di rete o lo sbilanciamento annullano il gather parallelo,
la modalità deve restare opzionale.

## Limiti e sicurezza

- il backbone completo deve stare sul coordinator secondo il profilo locale;
- ogni worker riceve attivazioni del modello in chiaro;
- la perdita di un peer invalida gli esperti che possiede;
- chat e benchmark non possono condividere simultaneamente la route;
- il protocollo non offre autenticazione o TLS.

Usare soltanto una rete fidata. Per il quadro completo vedere
[INFERENZA-DISTRIBUITA.md](INFERENZA-DISTRIBUITA.md) e
[CRITTOGRAFIA.md](CRITTOGRAFIA.md).

## Mappa del codice

- `Sources/DS4Engine/Distributed/Protocol/Experts`
- `Sources/DS4Engine/Distributed/Coordinator/DistCoordinator+ExpertParallelism.swift`
- `Sources/DS4Engine/Distributed/Coordinator/DistCoordinator+VerticalChat.swift`
- `Sources/DS4Engine/Distributed/Execution/ExpertShard.swift`
- `Sources/DS4Engine/Distributed/Worker/Assignments/DistWorker+ExpertAssignment.swift`
- `Sources/DS4Metal/Decode/Execution/StreamingDecoder.swift`
- `Sources/DS4Metal/Decode/Execution/DecodeLayer.swift`
