# DwarfStar/App

Application shell and shared state.

- **`DwarfStarApp.swift`** is the `@main` entry point. It creates `ChatStore` and
  the main window.
- **`RootView.swift`** owns the `NavigationSplitView` sidebar and instantiates the
  shared controllers for Chat, Distributed, Server, Bench, and Diagnostics. The
  Server and Bench controllers receive the `ChatStore`/shared engine instead of
  loading independent full engines.
- **`AppSettings.swift`** stores persisted settings such as model path and context
  length plus local/distributed execution mode. Settings owns these values;
  other controllers read them through the shared app state.
- **`AppEnvironment.swift`** resolves paths for development vs bundled app runs,
  computes RAM-based hardware presets, and exposes memory helpers.
