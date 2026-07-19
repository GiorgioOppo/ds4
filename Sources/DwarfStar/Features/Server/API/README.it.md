[English](README.md) | **Italiano**

# Adapter API del Server

Adapter specifici per endpoint traducono il JSON HTTP in richieste per
l'engine e rimandano in streaming l'output dell'engine nel formato wire
richiesto.

- `ChatRequestParser.swift`: parsing condiviso dei messaggi e dei parametri di
  generazione.
- `LocalServer+OpenAIChat.swift`: `/v1/chat/completions`.
- `LocalServer+Responses.swift`: `/v1/responses`.
- `LocalServer+LegacyCompletions.swift`: `/v1/completions`.
- `LocalServer+Anthropic.swift`: `/v1/messages`.

Gli adapter possono dipendere da `LocalServer`, `DS4Core` e `DS4Engine`; non
devono aprire socket direttamente. Mantieni espliciti i default di
compatibilità, riporta il modello effettivamente caricato e instrada tutta la
generazione attraverso il request gate.
