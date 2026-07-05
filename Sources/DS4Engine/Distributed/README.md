# DS4Engine/Distributed

Distributed inference with pipeline parallelism across contiguous layer ranges,
modeled after `ds4_distributed.c`.

Each **worker** owns a slice of layers, including weights and its KV shard. The
**coordinator** owns embeddings, sampling, prompt rendering, and cluster control.
For each token, the HC state (`nHC x nEmbd` floats, transported as 32/16/8-bit
depending on configuration) moves through the worker chain.

The COORDINATOR defines each worker's job (protocol v3): workers start IDLE —
listening, no model loaded — and at connect time the coordinator splits the
layers across the peer list (in order; the last slice also runs the output
head) and sends each worker an ASSIGN with the gguf, the context size, the
expert-cache budget, and the slice to own. The worker loads — or reuses — its
engine and replies READY.

The coordinator also DISTRIBUTES the files (protocol v5): workers need no
local gguf. After HELLO the coordinator sends a FILE OFFER (name, size,
SHA-256 for the gguf and, when enabled, the expert-bundle sidecar); the worker
answers with what it is missing — checking its managed store
(`Application Support/DwarfStar/dist-models`, manifest hashes recorded at
reception) and hash-matching local files — and only the missing files are
streamed (sequential 4 MB chunks, F_NOCACHE on both ends, hash accumulated
inline and verified at DONE). The huge setup runs ONCE: later connects answer
the offer from cached manifests in milliseconds. Hashing a coordinator-side
gguf is also one-time (persistent size+mtime-validated hash cache). ASSIGN
carries `useExpertBundle`, so the sidecar decision is the coordinator's too.

KV continuity (protocol v4) removes the per-turn full re-prefill:

- **In-memory prefix reuse**: the coordinator tracks the token prefix committed
  by the last CLEAN turn; when the next re-rendered conversation extends it
  exactly, only the suffix is prefilled (the shards still hold that KV). Any
  mismatch, Stop, error, or benchmark falls back safely (dirty-until-clean,
  like the local engine — the NSA compressor cannot rewind).
- **Per-shard disk checkpoints**: with the coordinator's disk-KV setting on,
  each worker keeps a slice-keyed `DiskKVStore` (budget from ASSIGN). After a
  clean turn the coordinator broadcasts `kvSave` (export under the compute
  gate, F_NOCACHE write in background); on a cold start it negotiates
  `kvQuery`/`kvRestore`: the stored prefix lengths of every shard are
  intersected and the longest COMMON one is restored everywhere — any shard
  missing it fails the negotiation and the turn cold-prefills.
- **Usage imatrix on workers**: ASSIGN carries the coordinator's richest local
  profile for the model; the worker prefers its own slice-refined profile
  (persisted between turns and on stop) and pre-warms its expert slot-cache.

- **`DistEngine.swift`** is the per-node engine. It exposes low-level slice
  operations (`embed`, `forwardSlice`, `head`) plus tokenizer/sampling utilities
  needed by the coordinator.
- **`DistCoordinator.swift`** connects workers, validates the protocol version,
  partitions the layers and ASSIGNs each worker its job, validates contiguous
  coverage, runs multi-turn chat on the cluster, and exposes `benchmark()`.
  Every turn gets a fresh `session` id that workers echo in each RESULT: a
  reply left in a TCP buffer by a cancelled turn is discarded on the next read
  instead of being mistaken for the new turn's answer.
- **`DistWorker.swift`** implements the worker node: it starts idle, loads its
  slice engine on ASSIGN (reusing it when the assignment is unchanged), and
  executes WORK frames. Frames are validated (token count vs payload, slice
  match, position vs context) before touching the engine, and one TURN at a
  time is enforced at the session level — a competing coordinator gets an
  explicit ERROR frame instead of silently resetting the active turn's KV shard.
- **`DistProtocol.swift` / `DistTransport.swift`** define protocol frames and the
  async `NWConnection` transport. Decoding is STRICT: truncated activation
  payloads and hostile lengths (route cap) reject the frame rather than
  producing short arrays. The activation codec moves data as bulk buffer
  copies — it is the wire hot path. TCP traffic is plaintext and listeners are
  unauthenticated, so use it only on trusted networks.
