# DwarfStar/Distributed

UI dell'inferenza distribuita (il motore sta in `DS4Engine/Distributed`).

- **`DistributedController.swift`** — guida sia il ruolo **worker** (questo Mac possiede uno slice di layer) sia il **coordinatore** (in Chat → Distribuito: connette i worker ed esegue la chat sul cluster). Espone la connessione al pannello Benchmark.
- **`DistributedView.swift`** — il pannello Worker (slice di layer, porta, log).
