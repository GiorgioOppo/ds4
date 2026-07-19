[English](README.md) | **Italiano**

# DwarfStar/Features/Server

Server HTTP nativo in-process costruito su `Network.framework`, compatibile
con le API in stile OpenAI e Anthropic. Non lancia un sottoprocesso e non
carica un secondo modello. Il server avvolge l'**unica `InferenceService`
condivisa** caricata nelle Impostazioni, quindi le richieste Chat e HTTP usano
lo stesso attore-motore e vengono serializzate una alla volta. Questo evita di
duplicare il Q4 residente, i buffer `mlock`ati, lo scratch KV e la memoria
della cache degli esperti sui sistemi da 16 GB.

Non c'è scelta del modello via HTTP: `/v1/models` pubblicizza esattamente il
modello caricato (il basename del GGUF); un campo `model` diverso in una
richiesta viene loggato e sovrascritto, e ogni risposta riporta il modello
realmente caricato.

- **`Controllers/ServerController.swift`** avvia/ferma il listener e lo lega
  al motore condiviso già caricato.
- **`Services/LocalServer.swift`** possiede lo stato condiviso del server;
  estensioni mirate sotto `API/` e `Networking/` implementano il routing degli
  endpoint per `/v1/chat/completions`, `/v1/responses`, `/v1/completions`,
  `/v1/messages` e `/v1/models`, incluse le risposte streaming e
  non-streaming.
- **`API/ChatRequestParser.swift`** interpreta i body delle richieste in tipi
  a livello di motore.
- **`Concurrency/RequestGate.swift`** serializza l'accesso al motore
  condiviso.
- **`Views/ServerView.swift`** disegna il pannello del server.

Limiti nella gestione delle richieste: i body sono limitati a 32 MB (413
oltre quella soglia) e un client deve consegnare l'intera richiesta entro
60 s, altrimenti la connessione viene chiusa — nessuno dei due casi può
bloccare il motore serializzato, perché il gate viene acquisito solo dopo che
la richiesta è stata letta e interpretata per intero.

Nel pannello del server si può impostare una API key opzionale: quando è
presente, ogni richiesta `/v1` deve inviare `Authorization: Bearer <key>`
(stile OpenAI) oppure `x-api-key: <key>` (stile Anthropic); qualsiasi altra
cosa riceve un 401.

Il traffico HTTP è in chiaro. Il default previsto è `127.0.0.1`; se lo esponi
oltre il loopback, mettilo dietro TLS — la API key protegge dagli altri
processi locali, non dall'intercettazione sulla rete.

Default del pannello: host `127.0.0.1`, porta `8000`, max token `1024`, CORS
disattivato, nessuna API key. Vedi il
[Riferimento di configurazione](../../../../README.it.md#riferimento-di-configurazione)
nella radice.

Vedi [`HTTP-API.md`](HTTP-API.it.md) per la matrice degli endpoint e il ciclo di
vita delle richieste. I README locali sotto `API/`, `Networking/`,
`Concurrency/`, `Services/`, `Controllers/` e `Views/` definiscono i confini
di proprietà. I nuovi endpoint dovrebbero riutilizzare gli helper di parsing,
autenticazione, risposta e request-gate invece di duplicarli.
