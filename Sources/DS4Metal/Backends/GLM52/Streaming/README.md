# GLM 5.2 streaming

First tranche of GLM roadmap step 1: from the validated read plan to real GGUF
payload bytes.

`GLM52PayloadReader` is bounded `pread` access to one GLM 5.2 payload. It reads
one validated descriptor in full (`read(_:into:)`, `bytes(of:)`) or executes a
`GLM52ExpertStreamPlan`, packing the eight selected experts as contiguous
gate|up|down records in router rank order (`read(plan:into:)`, layout exposed
by `packedLayout(of:)`).

Every read is proved twice before a byte moves: the planner/schema already
proved the range lies inside its tensor, and the reader re-proves it against
the real file size (`fstat`). The production constructor
(`init(path:weightMap:)`) rejects a truncated or substituted GGUF at open time
by checking the weight map's farthest descriptor against the file size. The 24
reads of one plan fill disjoint destinations concurrently
(`DispatchQueue.concurrentPerform`, explicit offsets — the
`GGUFWeights.gatherExperts` pattern); the serial path is byte-identical.

The gate|up|down record mirrors the interleaved slot layout of the DeepSeek
expert pool, so a future GLM slot cache can consume these buffers without
repacking.

The reader never interprets the quantized bytes it moves and owns no GPU
resource: it stays testable against synthetic files without a Metal device.
Any new read primitive must keep the double bounds proof (plan + file) and
typed errors that precede any partial fill.
