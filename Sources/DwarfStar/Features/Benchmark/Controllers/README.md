# Benchmark Controllers

`BenchController.swift` owns benchmark inputs, lifecycle, results, and error
state on the main actor. It dispatches either to the shared local
`InferenceService` or to the connected distributed coordinator.

The controller is the only place where a UI benchmark run may mutate engine KV
state. Keep the idle-chat gate intact, publish view-ready `BenchRow` values, and
never load a private model copy from this layer.

