import Foundation
import DS4Core

/// Wire protocol for DwarfStar distributed inference (pipeline parallelism by
/// contiguous layer ranges), modelled on ds4_distributed.c but Swift-native:
/// every node runs the same DwarfStar build, so the framing is our own (no C
/// byte-compatibility needed).
///
/// Topology: a COORDINATOR owns the embedding, the sampling loop and the API/UI.
/// Each WORKER owns a contiguous layer slice `[start, end]` and its KV/compressor
/// shard. Per token the coordinator embeds → sends the hidden (HC) state to the
/// first worker → each worker runs its layers and forwards the HC state → the
/// coordinator runs the output head and samples. The HC state is `nHC*nEmbd`
/// floats; it can be transported at 32/16/8-bit width to save bandwidth.
public enum Dist {
    /// Bump when the framing or semantics change incompatibly.
    /// v2: HELLO carries the version; WORK/RESULT carry a per-turn `session`
    /// id echoed by the workers, so a result left in a TCP buffer by a
    /// cancelled turn can never be mistaken for the next turn's reply.
    /// v3: the COORDINATOR defines each worker's job. Workers start idle
    /// (listening, no model loaded); the coordinator sends ASSIGN (gguf,
    /// context, layer slice, cache slots) and the worker replies READY once
    /// its engine is loaded. HELLO gained the `assigned` state.
    /// v4: distributed KV continuity. ASSIGN carries the usage imatrix (slot
    /// cache pre-warm) and a disk-KV token budget; new KV control frames let
    /// the coordinator checkpoint/restore each worker's shard (kvQuery /
    /// kvLengths / kvRestore / kvSave / kvAck); WORK gained `turnStart` so a
    /// turn can begin mid-context (restored or reused prefix).
    /// v5: the coordinator DISTRIBUTES the files. Workers no longer need a
    /// local gguf: after HELLO the coordinator sends a FILE OFFER (name, size,
    /// sha256 for gguf + sidecar); the worker answers with what it is missing
    /// (hash-verified against its managed store and its local files) and the
    /// coordinator streams only those — the huge setup runs ONCE; later
    /// connects verify hashes from cached manifests in milliseconds.
    /// v6: derived caches travel too — the offer can include the Q4 dense
    /// requant cache (<gguf>.q4dense, ~1.4 GB beats minutes of re-requant on
    /// every worker) and ASSIGN carries the Q4 on/off decision.
    /// v7: WORK carries the chunk's token ids. The first n_hash_layer (3)
    /// layers route experts by TOKEN ID (ffn_gate_tid2eid), so a shard that
    /// covers them cannot route from the HC state alone.
    /// v8: RESUMABLE transfers. The offer carries a CHAINED-HASH checkpoint
    /// list per file (one SHA-256 every fileCheckpointBytes, each folded over
    /// the previous — chain[k] commits to the whole prefix); the worker keeps
    /// its `.part` across disconnects AND sessions, validates it block-by-block
    /// against the chain, truncates to the last good checkpoint and answers
    /// FILE NEED with a per-file RESUME OFFSET. The coordinator streams from
    /// there and retries a broken peer setup up to 3 times before failing.
    /// v9: ASSIGN carries the coordinator's PERFORMANCE KNOBS (DS4_* env,
    /// whitelisted). A worker with factory defaults ran the engine without
    /// dense streaming/mlock/pread — measured 0.37 tok/s against the same
    /// hardware's 2.7 local. The coordinator's measured configuration IS the
    /// job definition: the worker applies it before loading its engine.
    /// v10: EXPERT PARALLELISM (scissione verticale, Fase B/C). Nuovi frame
    /// expertAssign/expertWork/expertSum: i worker possono servire shard di
    /// ESPERTI (mask sui 256, tutti i layer) e il coordinatore può montare la
    /// route verticale (backbone denso locale + FFN routed remota). Il bump
    /// evita il mix: un worker più vecchio ignorerebbe l'expertAssign e il
    /// coordinatore resterebbe in attesa del READY per sempre.
    /// v11: geometria DeepSeek V4 ricavata dal GGUF (Flash 43/256 e Pro
    /// 61/384). EXPERT ASSIGN prefissa la mask con la sua lunghezza invece di
    /// fissarla a 32 byte; READY deve riportare il numero di layer realmente
    /// caricato e ogni slice viene convalidata contro quel modello.
    public static let protocolVersion: UInt32 = 11

    /// The env knobs an ASSIGN may carry and a worker will apply (v9). A
    /// WHITELIST on both sides: the wire must never gain the power to set
    /// arbitrary environment on a worker. All performance-only — none of
    /// these can change the numerics of the shard (DS4_DENSE_Q4, the one
    /// lossy knob, travels as a typed field and its cache as a file).
    public static let perfKnobKeys: [String] = [
        "DS4_DENSE_STREAM", "DS4_DENSE_AHEAD", "DS4_MLOCK",
        "DS4_EXPERT_PREAD", "DS4_PREAD_SPLIT", "DS4_WILLNEED_EXPERTS",
        "DS4_ASYNC_FFN", "DS4_EXPERT_LOOKAHEAD", "DS4_EXPERT_ASYNC_SPLIT",
        "DS4_Q8_NSG",
        "DS4_LAZY_IDX", "DS4_RESIDENT_COMP", "DS4_FUSED_HC", "DS4_FUSED_MOE",
        "DS4_RAW_RING", "DS4_EXPERT_CACHE_UNIFORM", "DS4_POOL_INTERLEAVE",
        "DS4_PREFILL_UNION", "DS4_PREFILL_CHUNK", "DS4_PREFILL_FFN_BATCH",
        "DS4_PREFILL_ROUTE_BATCH", "DS4_PREFILL_MM",
    ]
    static let magic: UInt32 = 0x44_53_34_44   // "DS4D"
    /// Sanity cap on the route length in a WORK frame (a hostile frame could
    /// otherwise declare 4G entries and spin the decoder).
    static let maxRouteEntries = 256
    /// Cap on entries in a FILE OFFER / NEED (gguf + a handful of sidecars).
    static let maxFileEntries = 16
    /// File-transfer chunk payload (per fileChunk frame).
    public static let fileChunkBytes = 4 * 1024 * 1024
    /// Checkpoint granularity of the chained-hash list (v8 resumable
    /// transfers): one 32-byte digest every 256 MB — a 70 GB gguf carries
    /// ~280 digests (~9 KB) in the offer, and at most 256 MB are re-sent
    /// after an interruption. MUST be a multiple of fileChunkBytes.
    public static let fileCheckpointBytes: UInt64 = 256 * 1024 * 1024

    public enum MsgType: UInt32, Sendable {
        case hello     = 1    // worker → coordinator on connect (state + model identity)
        case work      = 3    // coordinator → worker: embed/HC input for a layer slice
        case result    = 4    // worker → coordinator: HC state or logits
        case error     = 2
        case assign    = 5    // coordinator → worker: gguf + settings + layer slice
        case ready     = 6    // worker → coordinator: assignment loaded (HELLO payload)
        case kvQuery   = 7    // coordinator → worker: which stored prefixes of these ids?
        case kvLengths = 8    // worker → coordinator: stored prefix lengths (may be empty)
        case kvRestore = 9    // coordinator → worker: restore EXACTLY these tokens' checkpoint
        case kvSave    = 10   // coordinator → worker: checkpoint your shard for these tokens
        case kvAck     = 11   // worker → coordinator: kvRestore/kvSave outcome
        case fileOffer = 12   // coordinator → worker: manifest (name+size+sha256 per file)
        case fileNeed  = 13   // worker → coordinator: offer indices it is missing
        case fileChunk = 14   // coordinator → worker: sequential slab of one file
        case fileDone  = 15   // coordinator → worker: file complete (worker verifies hash)
        case fileAck   = 16   // worker → coordinator: per-file receive outcome
        case progress  = 17   // worker → coordinator: load progress text (informational)
        // Expert parallelism (scissione VERTICALE — vedi
        // docs/EXPERT_PARALLELISM.md). Questi frame alimentano il percorso
        // coordinator/backbone + worker expert-shard attivo dal protocollo v10.
        case expertAssign = 18  // coordinator → worker: shard di ESPERTI (mask), non di layer
        case expertWork   = 19  // coordinator → worker: attivazione + id/pesi del layer corrente
        case expertSum    = 20  // worker → coordinator: somma parziale pesata degli esperti
    }

    /// Flags on a WORK message.
    public struct WorkFlags: OptionSet, Sendable {
        public let rawValue: UInt32
        public init(rawValue: UInt32) { self.rawValue = rawValue }
        public static let resetSession = WorkFlags(rawValue: 1 << 0)  // pos==0: reset compressor/KV
        public static let outputLogits = WorkFlags(rawValue: 1 << 1)  // last slice: this worker also runs the head
        /// First chunk of a TURN (chat send or benchmark). Workers adopt the
        /// chunk's session id here — a turn may start mid-context (pos > 0)
        /// when the coordinator reuses or restores a KV prefix.
        public static let turnStart = WorkFlags(rawValue: 1 << 2)
    }

    public enum ResultKind: UInt32, Sendable { case hidden = 0, logits = 1, ack = 2 }
}
