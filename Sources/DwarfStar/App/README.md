# DwarfStar/App

Application shell and shared state.

- **`DwarfStarApp.swift`** is the `@main` entry point. It installs the
  `EngineLog` stderr capture, creates `AppSettings`, `ChatStore`, and `MCPStore`,
  and opens the main window.
- **`RootView.swift`** owns the `NavigationSplitView` sidebar and instantiates the
  shared controllers for Distributed, Server, Bench, and Diagnostics (the
  `ChatStore` is created in `DwarfStarApp` and passed in). The Server and Bench
  controllers receive the `ChatStore`/shared engine instead of loading
  independent full engines.
- **`AppSettings.swift`** stores persisted settings such as model path
  (`DS4ModelPath`), context length (`DS4ContextSize`), and local/distributed
  execution mode (`DS4EngineMode`). Settings owns these values; other
  controllers read them through the shared app state. See the root
  [Configuration Reference](../../../README.md#configuration-reference) for all
  keys and defaults.
- **`AppEnvironment.swift`** resolves paths for development vs bundled app runs,
  computes RAM-based hardware presets (default context 4096 below 24 GB, 8192
  below 80 GB, 32768 above), and exposes memory helpers.

## Change rules

- Create process-wide observable state here and inject it into features.
- Keep feature-specific state in that feature's controller or view model.
- Preserve the single shared local engine and the main-actor ownership of UI
  state.
- Add new sidebar destinations through `AppSection` and `RootView`, with their
  implementation under `Features/`.
