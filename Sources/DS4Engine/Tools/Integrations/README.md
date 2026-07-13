# Tools/Integrations

Contiene implementazioni con side effect riusate dai built-in.

## Componenti

- `WebClient.swift`: HTTP(S), protezione SSRF, redirect, timeout, limite body e
  conversione HTML in testo.
- `GitTool.swift`: sottoinsieme whitelisted di comandi git locali.
- `GitHubTool.swift`: download controllato di repository pubblici come archivio,
  estrazione e import in `ProjectCache`.

## Flusso e dipendenze

I built-in in [`Web`](../Builtins/Web/README.md) e `Git.swift` traducono JSON in
chiamate a queste integrazioni. Le integrazioni restituiscono dati o errori e non
formattano il prompt del modello.

## Estensione

Centralizzare qui validazione di rete/processo e limiti condivisi. Non esporre
shell arbitraria, host privati o argomenti non convalidati. Ogni nuova
integrazione esterna deve supportare cancellazione e produrre output bounded.
