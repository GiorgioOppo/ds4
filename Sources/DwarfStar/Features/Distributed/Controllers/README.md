**English** | [Italiano](README.it.md)

# Distributed Controllers

`DistributedController.swift` owns coordinator and worker lifecycle, connection
configuration, logs, and the coordinator reference shared with Chat and
Benchmark. It adapts `DS4Engine` distributed APIs to main-actor UI state.

Keep network protocol, framing, file transfer, and execution logic in
`DS4Engine/Distributed`. Changes here should only coordinate those APIs and
must preserve orderly stop/disconnect cleanup.

