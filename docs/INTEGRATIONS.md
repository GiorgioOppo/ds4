# DeepSeekIntegrations — Adapter per Sistemi Esterni

Adapter opzionali che collegano il toolbox DeepSeekTools a sistemi esterni. Tenuti in un target separato per non propagare le loro dipendenze al core.

## Moduli

```
DeepSeekIntegrations/
├── README.md                    # Documentazione integrazioni
├── HTTPRecorder/               # Registrazione traffico HTTP
├── Sandbox/                    # Sandbox macOS per tool shell
└── Slack/                      # Integrazione Slack (scaffold)
```

## HTTPRecorder

Middleware per registrare richieste HTTP effettuate dai tool di rete (WebFetchTool, WebSearchTool) per debug e audit. Intercetta le chiamate al `URLSession` condiviso e salva richieste/risposte su disco.

## Sandbox

Helper per eseguire comandi shell in una sandbox macOS (`sandbox-exec`). Fornisce profili di sandboxing predefiniti per limitare l'accesso a filesystem, rete e processi durante l'esecuzione di tool shell.

## Slack

Scaffold per integrazione con Slack: invio messaggi, lettura canali, notifiche. Non ancora implementato.

## Stato Implementazione

| Integrazione | Stato |
|-------------|-------|
| HTTPRecorder | ✅ Scaffold funzionante |
| Sandbox | ✅ Scaffold funzionante |
| Slack | ❌ Stub