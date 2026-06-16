# DwarfStar/App

Shell dell'app e stato condiviso.

- **`DwarfStarApp.swift`** — `@main`: crea lo `ChatStore` e la finestra.
- **`RootView.swift`** — `NavigationSplitView` con la sidebar; istanzia e condivide i controller (Chat, Distribuito, Server, Bench, Diagnostica).
- **`AppSettings.swift`** — settings persistiti (path modello, contesto…) posseduti da Impostazioni e proxati dagli altri controller.
- **`AppEnvironment.swift`** — risoluzione dei path (dev vs bundle), preset hardware in base alla RAM, helper memoria.
