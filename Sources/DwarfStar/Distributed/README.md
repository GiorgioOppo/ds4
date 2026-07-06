# DwarfStar/Distributed

UI for distributed inference. The engine and protocol live under
`DS4Engine/Distributed`.

- **`DistributedController.swift`** drives both roles. As a **worker**, this Mac
  owns a slice of layers. As a **coordinator** from Chat -> Distributed, it
  connects workers and runs chat over the cluster. It also exposes the connected
  coordinator to the Benchmark panel.
- **`DistributedView.swift`** contains `WorkerView` (the Worker sidebar panel:
  listening port and logs — the layer slice, model, and settings are assigned by
  the coordinator at connect time) and `CoordinatorChatView` (the coordinator
  chat shown inside the Chat tab in Distributed mode).

Defaults: worker port 9100, peer list `127.0.0.1:9100`, activation bits 32,
prefill chunk 32, max tokens 512, return port 9099. See the root
[Configuration Reference](../../../README.md#configuration-reference).
