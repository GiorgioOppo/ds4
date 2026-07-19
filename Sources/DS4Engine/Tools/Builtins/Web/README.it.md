[English](README.md) | **Italiano**

# Builtins/Web

Espone al modello accesso web limitato.

## Tool

- `web_search`: interroga l'endpoint configurato (DuckDuckGo di default) e
  restituisce titolo, URL e snippet.
- `web_fetch`: scarica HTTP(S), converte HTML in testo e supporta finestre tramite
  `offset` per pagine lunghe.

## Flusso e dipendenze

Entrambi delegano a [`WebClient`](../../Integrations/README.it.md), che applica
protezione SSRF, validazione dei redirect, timeout e limite del body. L'endpoint
di ricerca può essere impostato con `DS4_SEARCH_URL` e deve contenere `%@`.

## Estensione

Non usare `URLSession` direttamente nel built-in. Preservare i limiti di output
e non seguire URL estratti da contenuto remoto senza la stessa validazione.
macOS ATS può rifiutare pagine HTTP non sicure: HTTPS è il percorso previsto.
