[English](README.md) | **Italiano**

# DwarfStar/Shared/Support

Utility GUI trasversali.

- **`EngineLog.swift`** memorizza un buffer di coda dei log dell'engine,
  mostrato dopo gli errori di chat per dare all'utente contesto immediato.
- **`ProcessStream.swift`** fornisce helper per percorsi assoluti e stream di
  output dei sottoprocessi.

Questi helper sono infrastruttura del layer applicativo e non possono
dipendere da una feature specifica. Mantieni limitata la cattura dei log,
preserva la separazione stdout/stderr e sposta il comportamento di dominio in
`DS4Core` o `DS4Engine`.
