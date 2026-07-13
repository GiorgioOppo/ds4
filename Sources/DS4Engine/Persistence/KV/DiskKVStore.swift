import Foundation
import DS4Core
import DS4Metal

/// Disk-backed KV cache, modelled on ds4_kvstore.c: completed-generation
/// checkpoints are written to a directory keyed by their exact token prefix;
/// a later conversation (or a stateless HTTP request re-sending the transcript)
/// that starts with a stored prefix RESTORES it and prefills only the rest.
///
/// File layout (Swift-defined body behind the ported KVC/DSV4 headers):
///   [48B KVC header]  (KVCFile: magic, quant, tokens, ctx, hits, timestamps)
///   [u32 nameLen][model name utf8]
///   [u32 nTokens][nTokens × u32 token ids]
///   [52B DSV4PayloadHeader]
///   per layer: [u32 rawStart][u32 rawFloats][raw f32…]
///              [u8 hasComp]( [u32 count][u32 stateLen][stateKv f32…]
///                            [stateScore f32…][u32 cacheFloats][cache f32…] )
/// Eviction uses the ported `KVCFile.evictionScore` under a byte budget;
/// hits/lastUsed are bumped in-place on every restore (the 48B header only).
///
/// RAM discipline, both directions: restore STREAMS the file one layer at a
/// time into the decoder (each batch is parsed, imported, and freed before the
/// next is read — peak = one layer, never the whole checkpoint), and store
/// writes from a uniquely-owned snapshot whose layers are dropped as they hit
/// the disk. Both sides read/write with F_NOCACHE so checkpoint bytes never
/// displace the hot page cache (dense weights / expert bundle).
public final class DiskKVStore: @unchecked Sendable {
    public struct Options: Sendable {
        /// Don't checkpoint tiny prefixes (C default is 512; local chats have
        /// shorter useful prefixes, so we default lower).
        public var minTokens = 128
        /// Re-checkpoint only after this many NEW tokens since the last store.
        public var storeIntervalTokens = 256
        public init() {}
    }

    public let directory: URL
    public let options: Options
    let budgetBytes: UInt64
    /// Total TOKEN budget across all stored entries (0 = byte budget only).
    /// Tokens are the natural unit here — "keep up to 1M tokens of checkpoints"
    /// — and per-token bytes vary with the model, so eviction counts tokens.
    let budgetTokens: Int
    let quantBits: UInt8
    let contextSize: Int

    public init(directory: URL, budgetMB: Int, quantBits: UInt8, contextSize: Int,
                budgetTokens: Int = 0, options: Options = Options()) throws {
        self.directory = directory
        self.budgetTokens = max(0, budgetTokens)
        // Token-budgeted stores derive a generous byte SAFETY cap from the token
        // budget (~32 KB/token upper bound incl. per-entry overhead) so an
        // unexpected entry mix still can't grow the directory without bound.
        self.budgetBytes = budgetTokens > 0
            ? UInt64(budgetTokens) * 32_768
            : UInt64(max(64, budgetMB)) * 1_048_576
        self.quantBits = quantBits == 2 ? 2 : 4    // header validity wants {2,4}
        self.contextSize = contextSize
        self.options = options
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

}
