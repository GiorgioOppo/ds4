# API HTTP locale

Il server in-process espone il modello già caricato dall'app; non crea mai un
secondo servizio di inferenza.

## Endpoint

| Endpoint | Livello di compatibilità |
|---|---|
| `GET /v1/models` | Scoperta del modello caricato |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/completions` | OpenAI Completions legacy |
| `POST /v1/messages` | Anthropic Messages |

Le richieste in streaming e non-streaming usano la stessa sorgente di eventi
dell'engine. Il server riporta il basename effettivo del GGUF anche quando una
richiesta fornisce un altro identificatore di modello.

## Percorso della richiesta

1. `Network.framework` accetta e delimita la richiesta.
2. Il parsing di `HTTPRequest` valida gli header e il framing del body.
3. I controlli su API key ed endpoint rifiutano le richieste non valide prima
   della generazione.
4. L'adapter dell'endpoint converte il JSON nei tipi di conversazione
   dell'engine.
5. `RequestGate` serializza l'inferenza rispetto alla Chat e agli altri client
   HTTP.
6. Gli helper di risposta emettono JSON o frame di streaming e chiudono la
   connessione.

Il listener predefinito è limitato al loopback e in chiaro. Il binding a
un'altra interfaccia richiede un reverse proxy TLS esterno; l'API key opzionale
non fornisce cifratura del trasporto.
