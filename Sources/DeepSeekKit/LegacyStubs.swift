import Foundation

// MARK: - Legacy KV Cache Stubs

public struct KVCacheSnapshot: @unchecked Sendable {
    public enum DiskCompression: Sendable {
        case f16, bf16, f32
    }
    public var totalBytes: Int { return 0 }
    public static func load(from url: URL) -> KVCacheSnapshot? { return nil }
    public func save(to url: URL, compression: DiskCompression) throws {}
}

public struct KVCacheFile: Sendable {
    public static let manifestMagic: UInt32 = 0x4B56434D
    public static func coldSaveAlignedCount(_ count: Int) -> Int { return count }
}

// In MLX, we don't snapshot explicit KV buffers this way for now.
public extension Transformer {
    func snapshotKVCache() -> KVCacheSnapshot { return KVCacheSnapshot() }
    func canRestoreKVCache(_ snap: KVCacheSnapshot) -> Bool { return false }
    func restoreKVCache(_ snap: KVCacheSnapshot) {}
}

public struct LoadPlan: Sendable {
    public enum Strategy: String, Sendable {
        case mlx_native = "MLX_NATIVE"
        case memoryMapped = "MEMORY_MAPPED"
    }

    public let physicalRAM: UInt64
    public let availableRAM: UInt64
    public let cores: Int
    public let mtlWorkingSet: UInt64
    public let shards: [String]
    public let totalBytes: UInt64
    public let maxShardBytes: UInt64
    public let strategy: Strategy
    public let reason: String

    public static func decide(modelDir: URL, override: String? = nil, forceLoad: Bool = false) throws -> LoadPlan {
        return LoadPlan(
            physicalRAM: ProcessInfo.processInfo.physicalMemory,
            availableRAM: ProcessInfo.processInfo.physicalMemory,
            cores: ProcessInfo.processInfo.activeProcessorCount,
            mtlWorkingSet: ProcessInfo.processInfo.physicalMemory,
            shards: ["mlx_weights"],
            totalBytes: 0,
            maxShardBytes: 0,
            strategy: .mlx_native,
            reason: "Using MLX native unified memory backend."
        )
    }
}
