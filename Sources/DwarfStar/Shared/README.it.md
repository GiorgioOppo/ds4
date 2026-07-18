# DwarfStar Shared

Qui risiede il supporto applicativo trasversale alle feature. Il codice in
`Shared/` deve essere utilizzabile da più di una feature e non deve dipendere
da una view o da un controller specifici di una feature.

- `Support/` contiene utility per i processi e per i log dell'engine.

Preferisci `DS4Core` o `DS4Engine` per la logica di dominio riutilizzabile.
Questa directory serve solo per adapter e helper a livello di app.
