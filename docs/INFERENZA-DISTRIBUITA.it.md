# Inferenza distribuita

DwarfStar supporta due topologie distinte sullo stesso protocollo TCP nativo:
pipeline orizzontale per intervalli di layer ed expert parallelism verticale.
Questo documento descrive il comportamento implementato dal protocollo v11.

## Quando usarla

La distribuzione può ridurre il lavoro SSD per nodo o parallelizzare il gather
degli esperti, ma aggiunge latenza e traffico di rete. Prima di configurarla:

- usare build identiche su tutti i Mac;
- preferire Ethernet diretta o bridge Thunderbolt;
- mantenere la rete fidata: trasporto e server sono in chiaro;
- verificare localmente modello e impostazioni su ogni classe di hardware.

## Topologia orizzontale: pipeline di layer

```text
coordinator -> worker 1 [layer 0...a]
            -> worker 2 [layer a+1...b]
            -> worker N [layer ...ultimo + head]
            -> coordinator -> sampling
```

Il coordinatore possiede rendering, tokenizer, embedding, sampling, strumenti e
stato della conversazione. Ogni worker possiede un intervallo contiguo di layer,
i relativi pesi e lo shard KV. Lo stato HC attraversa i worker per ogni token o
chunk di prefill.

Questa modalità riduce gli esperti letti da ogni SSD a circa una frazione dei
layer totali. Il tempo per token resta però vicino alla somma dei tempi dei
worker, più il trasporto. Il forwarding worker-to-worker evita alcuni ritorni
al coordinatore, ma richiede un indirizzo di ritorno raggiungibile.

## Topologia verticale: expert parallelism

```text
                 +-> worker 1: subset esperti di tutti i layer --+
backbone locale -+-> worker 2: subset esperti di tutti i layer --+-> somma
                 +-> worker N: subset esperti di tutti i layer --+
```

Il coordinatore esegue l'intero backbone denso: embedding, route/attention, KV,
compressori, FFN shared e output head. Gli esperti routed di ogni layer — 256
per Flash o 384 per Pro — sono partizionati fra i worker con mask esplicite.
Per ciascun layer routed il
coordinatore invia ai soli proprietari coinvolti attivazione, id e pesi; i
worker restituiscono una somma parziale.

Il gather degli esperti può avvenire in parallelo sugli SSD, ma il percorso
richiede circa un round-trip per layer routed. È adatto soltanto a collegamenti
con RTT inferiore a circa 1 ms. Su Wi-Fi la latenza di rete domina per
costruzione.

Stato attuale:

- protocollo `expertAssign`, `expertWork`, `expertSum` attivo;
- worker con `ExpertShard` e bundle/cache filtrati dalla mask;
- backbone locale collegato tramite callback `remoteExperts`;
- chat verticale e benchmark dedicato disponibili nella GUI;
- partizione corrente round-robin; il bilanciamento dalla usage imatrix resta
  un miglioramento possibile.

I dettagli prestazionali sono in
[EXPERT_PARALLELISM.md](EXPERT_PARALLELISM.md).

## Ciclo di connessione

1. Il worker avvia un listener senza caricare alcun modello.
2. Il coordinatore apre la connessione e valida magic e versione.
3. Offre GGUF e sidecar con nome, dimensione e SHA-256.
4. Il worker richiede soltanto file o suffissi mancanti.
5. Il coordinatore invia `ASSIGN` per una slice orizzontale oppure
   `EXPERT_ASSIGN` per uno shard verticale.
6. Il worker applica la whitelist dei knob, ispeziona la geometria del GGUF,
   convalida la slice o la mask, carica il motore assegnato e invia progressi e
   `READY` con 43 layer per Flash o 61 per Pro.
7. La route diventa disponibile soltanto quando tutti i peer necessari sono
   pronti e la copertura è valida.

Il setup dei peer procede in parallelo. Una disconnessione durante un file
transfer conserva il `.part`; alla riconnessione la catena di hash individua
l'ultimo checkpoint valido e riparte da lì.

## Protocollo v11

Il framing usa magic `DS4D`, header little-endian e payload con limiti espliciti.
La versione deve coincidere esattamente: non esiste negoziazione fra semantiche
incompatibili.

`EXPERT_ASSIGN` contiene una mask a lunghezza prefissata: 32 byte per Flash e
48 per Pro. Il decoder rifiuta lunghezze errate, bit di padding non nulli e
payload troncati. Un worker non assegnato annuncia zero layer; dopo `ASSIGN`,
`READY` deve coincidere con la geometria realmente caricata.

Le famiglie di messaggi sono separate per responsabilità:

- handshake e assegnazione;
- work/result della pipeline;
- checkpoint KV;
- offerta, richiesta, chunk e conferma dei file;
- assegnazione e lavoro degli expert shard;
- errori e avanzamento.

Le strutture wire vivono in `Distributed/Protocol` e non dipendono da socket,
coordinator o worker. `DistTransport` gestisce connessione e frame; coordinator
e worker applicano la semantica.

## Continuità KV

Nella pipeline orizzontale ogni worker salva soltanto i layer che possiede. Il
coordinatore può riusare un prefisso in memoria o negoziare un ripristino su
disco. Il restore è accettato solo se tutti gli shard possiedono lo stesso
prefisso; in caso contrario l'intera route riparte da prefill freddo.

Session id e `turnStart` impediscono che un risultato rimasto nel socket dopo
uno stop venga interpretato come risposta del turno successivo.

La topologia verticale mantiene KV e stato ricorrente sul backbone locale; i
worker expert sono stateless rispetto alla sequenza e servono richieste FFN.

## Trasferimento dei file

I worker usano uno store gestito sotto Application Support. Il protocollo può
trasferire:

- GGUF;
- expert bundle;
- cache Q4 dense;
- altri sidecar dichiarati nel manifest.

Per Pro è eseguibile il GGUF Q2 completo a file singolo. Il package Pro Q4
`Layers00-30`/`Layers31-output` può essere trasferito e scaricato, ma non è una
route valida: l'assemblaggio multi-shard non è implementato.

I chunk sono da 4 MiB. Ogni file ha SHA-256 finale e checkpoint concatenati
ogni 256 MiB. Un file derivato viene riutilizzato soltanto se il manifest e le
dimensioni corrispondono.

## Configurazione

| Impostazione | Pipeline | Verticale |
|---|---|---|
| elenco `host:port` | ordine delle slice | elenco degli shard |
| activation bits | 32/16/8 per stato HC | 32/16 per attivazioni e somme |
| prefill chunk | token per frame | prefill del backbone locale |
| forwarding | opzionale | non applicabile |
| cache esperti worker | cache della slice | cache degli esperti posseduti |
| expert bundle | consigliato | fortemente consigliato |

Il coordinatore propaga solo `Dist.perfKnobKeys`. Le variabili arbitrarie non
possono essere impostate via rete. Le opzioni che cambiano i numeri, come Q4
dense, viaggiano in campi tipizzati e con il relativo sidecar.

## Concorrenza e fallimenti

- Un worker serve una route alla volta tramite `DistGate`.
- Stop e cancellazione chiudono il turno al prossimo confine sicuro.
- Errori di versione, payload o copertura sono fatali per il setup.
- Errori di trasporto durante il setup possono essere ritentati.
- Un peer verticale mancante invalida la partizione degli esperti.
- Benchmark e chat non possono usare contemporaneamente la stessa route/KV.

## Sicurezza

Il protocollo non offre TLS né autenticazione. Prompt, token e attivazioni
viaggiano in chiaro. Usare soltanto una LAN fidata o un tunnel protetto e non
esporre la porta worker su Internet. Vedere
[CRITTOGRAFIA.md](CRITTOGRAFIA.md).

## Mappa del codice

- `Sources/DS4Engine/Distributed/Protocol` — dati wire e codec.
- `Sources/DS4Engine/Distributed/Transport` — connessioni e framing.
- `Sources/DS4Engine/Distributed/Coordinator` — setup, file, KV e chat.
- `Sources/DS4Engine/Distributed/Worker` — lifecycle e serving.
- `Sources/DS4Engine/Distributed/Execution` — slice decoder ed expert shard.
- `Sources/DwarfStar/Features/Distributed` — controller e viste.

## Verifica consigliata

1. test round-trip dei payload e dei limiti;
2. connessione loopback con file già presenti;
3. interruzione e ripresa di un file parziale;
4. parità locale/pipeline con 32 bit;
5. A/B 16 e 8 bit per qualità e rete;
6. benchmark verticale solo dopo avere misurato RTT e baseline locale.
7. per Pro, parità numerica e benchmark multi-Mac sul GGUF Q2 reale prima di
   considerare conclusa la validazione prestazionale.

Non confrontare topologie con modelli, cache o warm-up diversi.
