import Foundation
import MLX

public final class WeightLoader {
    public let directory: URL
    private var arrays: [String: MLXArray] = [:]
    public private(set) var missing: Set<String> = []
    
    public var streamingEnabled: Bool = false
    public var streamingSlotCount: Int { 0 }
    
    public var totalKnownNames: Int { arrays.count }
    public var allKnownNames: [String] { Array(arrays.keys) }
    public var shardCount: Int { 1 } // Simplified
    
    public static func discoverShards(in directory: URL) throws -> [(url: URL, byteCount: UInt64)] {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey])
        let candidates = contents
            .filter { $0.pathExtension == "safetensors" || $0.pathExtension == "safetensors.lz4" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        
        return try candidates.map { url in
            let res = try url.resourceValues(forKeys: [.fileSizeKey])
            let size = res.fileSize.map(UInt64.init) ?? 0
            return (url: url, byteCount: size)
        }
    }
    
    public init(directory: URL) throws {
        self.directory = directory
        let fm = FileManager.default
        let contents = (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        let candidates = contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        
        if candidates.isEmpty {
            throw NSError(domain: "WeightLoader", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "no .safetensors files in \(directory.path)"
            ])
        }
        
        for url in candidates {
            // Load using MLX
            do {
                let loaded = try MLX.loadArrays(url: url)
                for (k, v) in loaded {
                    arrays[k] = v
                }
            } catch {
                print("Failed to load \(url): \(error)")
            }
        }
    }
    
    
    public func load(_ name: String) throws -> Tensor? {
        if let arr = arrays[name] {
            let dt: DType
            switch arr.dtype {
            case .float32: dt = .f32
            case .float16: dt = .f16
            case .bfloat16: dt = .bf16
            case .int32: dt = .i32
            case .int8: dt = .i8
            case .int64: dt = .i64
            case .uint8: dt = .fp8E4M3 // Assumption for quantization types
            default: dt = .f32
            }
            return Tensor(array: arr, dtype: dt)
        }
        missing.insert(name)
        return nil
    }
    
    public func shape(of name: String) -> [Int]? {
        return arrays[name]?.shape
    }
    
    public func shape(ofAny candidates: [String]) -> [Int]? {
        for n in candidates {
            if let s = shape(of: n) { return s }
        }
        return nil
    }
    
    public func tryLoad(_ candidates: [String]) throws -> Tensor? {
        for n in candidates {
            if let t = try load(n) { return t }
        }
        for n in candidates { missing.insert(n) }
        return nil
    }
    
    public func ensureLayer(_ K: Int) {}
    public func ensureExperts(layer K: Int, indices: [Int]) {}
    public func prefetchLayer(_ layerIndex: Int) {}
    public func releaseLayer(_ layerIndex: Int) {}
    
    @discardableResult
    public func warmupAllShards(memoryGuardRatio: Double = 1.5) -> Bool {
        return true
    }
}
