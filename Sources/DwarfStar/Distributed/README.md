# DwarfStar/Distributed

UI for distributed inference. The engine and protocol live under
`DS4Engine/Distributed`.

- **`DistributedController.swift`** drives both roles. As a **worker**, this Mac
  owns a slice of layers. As a **coordinator** from Chat -> Distributed, it
  connects workers and runs chat over the cluster. It also exposes the connected
  coordinator to the Benchmark panel.
- **`DistributedView.swift`** renders the Worker panel, including layer slice,
  port, and logs.
