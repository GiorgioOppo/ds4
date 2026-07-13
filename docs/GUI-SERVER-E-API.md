# GUI, server locale e API

La GUI SwiftUI e il server HTTP sono due ingressi allo stesso motore. Entrambi
devono passare da `InferenceService`; nessuna feature deve creare una seconda
copia del modello per servire una richiesta.

## Struttura dell'app

`Sources/DwarfStar/App` contiene soltanto bootstrap, ambiente, impostazioni
globali e navigazione. Il comportamento è organizzato sotto
`Sources/DwarfStar/Features`:

| Feature | Responsabilità |
|---|---|
| `Chat` | sessioni, rendering, allegati, tool loop e generazione |
| `Settings` | modello, memoria, prestazioni, MCP e configurazione distribuita |
| `ModelManagement` | catalogo e download dei GGUF |
| `Project` | selezione e ispezione del progetto attivo |
| `Tuning` | cache esperti, profili e auto-tune |
| `Server` | listener HTTP e adattatori OpenAI/Anthropic |
| `Distributed` | coordinator, worker e benchmark verticale |
| `Benchmark` | throughput del motore condiviso |
| `Diagnostics` | tokenizer, template e stato operativo |

`Shared/Support` ospita solo helper usati da più feature. Un helper specifico
della chat o del server deve restare nella feature proprietaria.

## Dipendenze condivise

`AppEnvironment` costruisce e distribuisce gli oggetti di lunga durata. Il più
importante è `InferenceService`, actor che serializza accesso al decoder e allo
stato KV. Chat, server e benchmark possono presentare interfacce diverse, ma
non eseguono inferenze concorrenti sullo stesso decoder.

Questo vincolo protegge:

- buffer Metal e cache esperti;
- posizione KV e stato ricorrente NSA;
- memoria residente sui sistemi con poca RAM;
- ordine degli eventi di generazione.

## ChatStore

`ChatStore` è il view model `@MainActor` della chat. Il file principale contiene
stato osservabile e inizializzazione; le estensioni separano:

- lifecycle del modello;
- generazione e cancellazione;
- sessioni persistenti;
- allegati;
- agenti e tool loop;
- benchmark e tuning;
- applicazione delle impostazioni prestazionali.

Il view model converte gli eventi del servizio in modelli di presentazione. Non
implementa tokenizer, sampler o parsing wire. Le viste non devono chiamare
direttamente il backend Metal.

## Server locale

`LocalServer` usa `Network.framework` ed espone il motore già caricato. Le
responsabilità sono divise in:

- `Networking` — accept, lettura HTTP, JSON e risposte;
- `API` — adattatori dei diversi contratti;
- `Concurrency` — limite e serializzazione delle richieste;
- `Services` — stato condiviso del listener;
- `Controllers` e `Views` — controllo dalla GUI.

Endpoint implementati:

| Metodo | Percorso | Compatibilità |
|---|---|---|
| `GET` | `/v1/models`, `/v1/models/{id}` | OpenAI |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions |
| `POST` | `/v1/responses` | OpenAI Responses |
| `POST` | `/v1/completions` | OpenAI legacy |
| `POST` | `/v1/messages` | Anthropic Messages |

Gli adattatori traducono la richiesta in tipi di inferenza condivisi e
riconvertono gli eventi in JSON o SSE. Non devono duplicare il loop di
generazione.

## Streaming SSE

Per una risposta streaming il server:

1. valida metodo, percorso, autenticazione e limite del body;
2. costruisce la richiesta normalizzata;
3. acquisisce il gate del motore;
4. invia header `text/event-stream`;
5. traduce ogni evento del servizio nel formato dell'API scelta;
6. invia il terminatore previsto dal protocollo;
7. libera il gate anche in caso di cancellazione o errore.

La disconnessione del client deve propagare la cancellazione al task di
generazione, senza lasciare un turno parzialmente committato come KV valido.

## Parametri e precedenza

Il server ha impostazioni proprie per host, porta, API key, CORS e limite
predefinito dei token. Sampling e limite presenti nel body della richiesta
sovrascrivono i default del server per quel turno. Il nome modello richiesto è
un identificatore di compatibilità: il server usa sempre il singolo GGUF già
caricato.

La tabella completa dei campi accettati è nella
[Configuration Reference](../README.md#http-server-server-tab).

## Sicurezza

Il listener usa HTTP in chiaro. Il default `127.0.0.1` limita l'accesso alla
macchina locale; un bind su LAN deve essere protetto esternamente con TLS o
tunnel. L'API key applicativa evita richieste accidentali ma, senza TLS, non
protegge il token da intercettazione.

Il body è limitato e il parser rifiuta richieste malformate. CORS è disattivato
di default. Vedere [CRITTOGRAFIA.md](CRITTOGRAFIA.md).

## Aggiungere una feature GUI

1. Creare `Features/<Nome>` con modelli, controller/view model, servizi e viste
   soltanto quando servono davvero.
2. Aggiungere un `README.md` che dichiari proprietà e dipendenze.
3. Esporre dal motore un'API applicativa in `DS4Engine`, non importare
   `DS4Metal` nella GUI.
4. Registrare la feature nella navigazione radice.
5. Persistire solo impostazioni che devono sopravvivere al riavvio.
6. Testare le parti pure nel target test e verificare manualmente il lifecycle
   SwiftUI che dipende dal sistema.

## Aggiungere o modificare un endpoint

1. Mantenere parsing HTTP generico in `Networking`.
2. Aggiungere il mapping del contratto sotto `API`.
3. Normalizzare verso i tipi di `DS4Engine/Inference/API`.
4. Coprire stream e non-stream, errori e cancellazione.
5. Non introdurre un decoder o una coda di sampling parallela.
6. Aggiornare questa guida, il README del server e gli esempi del README
   principale.

## File principali

- `Sources/DwarfStar/App/AppEnvironment.swift`
- `Sources/DwarfStar/Features/Chat/ViewModels/ChatStore.swift`
- `Sources/DwarfStar/Features/Server/Services/LocalServer.swift`
- `Sources/DwarfStar/Features/Server/Concurrency/RequestGate.swift`
- `Sources/DS4Engine/Inference/Service/InferenceService.swift`

Per il flusso interno vedere [PIPELINE-INFERENZA.md](PIPELINE-INFERENZA.md).
