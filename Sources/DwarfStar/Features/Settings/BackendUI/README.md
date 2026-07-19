**English** | [Italiano](README.it.md)

# Settings/BackendUI

Per-backend UI layer: an "abstract" base class plus one concrete
implementation per backend, selected from the loaded (or inspected) model.

## Main files

- [`BackendSettingsUI.swift`](BackendSettingsUI.swift): base class and
  factory. Implements the COMMON pieces — the benchmark scaffold
  (quick/full/stop/progress/report), the shared Disk KV rows and the
  fallback info section — and declares the override points (backend name,
  memory section, benchmark extras, tuning panel). Instantiable on purpose:
  the base is the neutral fallback for Qwen/unknown/no model.
- [`DeepSeekSettingsUI.swift`](DeepSeekSettingsUI.swift): DeepSeek V4
  implementation — full memory section (expert cache, dense streaming,
  lossy Q4, bundle, Disk KV with token budget, Raw-KV ring), auto-tune
  extras and the slot-cache + usage-imatrix tuning panel. Also hosts the
  shared `BundleBuildButton`.
- [`GLM52SettingsUI.swift`](GLM52SettingsUI.swift): GLM 5.2
  implementation — residency/experts/arena/streaming steppers, MetalIO and
  speculative staging toggles, Q4 sidecar, Disk KV without budget (single
  per-model checkpoint) and unified-sidecar build.

## Flow and dependencies

`SettingsView` and `TuningView` call `BackendSettingsUI.make(store:dist:)`
per body evaluation and render the returned sections. Section content binds
to `ChatStore` state through small `@Bindable` view structs, so SwiftUI
observation keeps working behind the class indirection.
`store.runSettingsBenchmark` and `store.buildExpertBundleNow` already
dispatch on the live backend, so the common scaffolds need no branching.

## Modification rules

Backend-specific controls belong in the backend's subclass, never in a
sibling's section or behind another backend's capability. New backends add
a subclass and one factory case. Keep the base sections generic: anything
mentioning a specific engine, knob or measured preset lives in the
subclass.
