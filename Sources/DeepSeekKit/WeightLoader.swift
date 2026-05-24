import Foundation
import MLX

// MARK: - Safetensors header types

/// Represents a single tensor's metadata from the safetensors header.
private struct TensorMeta {
    let dtype: String       // e.g. "F32", "BF16", "F16", "I8", "I32", "I64", "U8", "F8_E4M3"
    let shape: [Int]
    let offsetBegin: Int    // relative to data section start
    let offsetEnd: Int      // relative to data section start
    let shardURL: URL
    let dataOffset: Int     // absolute file offset = 8 + headerSize
}

/// Parses safetensors dtype string to our DType enum.
private func dtypeFromString(_ s: String) -> (DType, MLX.DType) {
    switch s.uppercased() {
    case "F32", "FLOAT32":      return (.f32, .float32)
    case "F16", "FLOAT16":      return (.f16, .float16)
    case "BF16", "BFLOAT16":    return (.bf16, .bfloat16)
    case "I32", "INT32":        return (.i32, .int32)
    case "I8", "INT8":          return (.i8, .int8)
    case "I64", "INT64":        return (.i64, .int64)
    case "U8", "UINT8":         return (.fp8E4M3, .uint8)  // quantization types stored as uint8
    case "F8_E4M3":             return (.fp8E4M3, .uint8)
    case "F8_E8M0":             return (.e8m0, .uint8)
    default:                    return (.f32, .float32)
    }
}

// MARK: - WeightLoader (Streaming)

public final class WeightLoader {
    public let directory: URL
    
    /// Index: tensor name → metadata (shape, dtype, file offset, shard).
    /// Built at init by parsing only the JSON headers — NO tensor data loaded.
    private var index: [String: TensorMeta] = [:]
    
    /// Cache of currently-loaded MLXArrays. Populated on-demand by
    /// `ensureLayer` and cleared by `releaseLayer`.
    private var cache: [String: MLXArray] = [:]
    private let cacheLock = NSLock()
    
    /// Set of tensor names that were requested but not found in any shard.
    private let missingLock = NSLock()
    public private(set) var missing: Set<String> = []
    
    /// Global tensors (embed, norm, head, hc_head_*) are always kept resident
    /// because they're used on every forward pass.
    private var globalNames: Set<String> = []
    
    public var streamingEnabled: Bool = true
    public var streamingSlotCount: Int { 2 }  // current + prefetch
    
    public var totalKnownNames: Int { index.count }
    public var allKnownNames: [String] { Array(index.keys) }
    public var shardCount: Int { shardURLs.count }
    
    private var shardURLs: [URL] = []
    
    // MARK: - Discovery
    
    public static func discoverShards(in directory: URL) throws -> [(url: URL, byteCount: UInt64)] {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey])
        let candidates = contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        
        return try candidates.map { url in
            let res = try url.resourceValues(forKeys: [.fileSizeKey])
            let size = res.fileSize.map(UInt64.init) ?? 0
            return (url: url, byteCount: size)
        }
    }
    
    // MARK: - Init (header-only — O(KB) RAM)
    
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
        
        self.shardURLs = candidates
        
        // Parse headers in parallel — only reads the first few KB of each file
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: candidates.count) { i in
            let url = candidates[i]
            do {
                let metas = try Self.parseHeader(url: url)
                lock.lock()
                for (name, meta) in metas {
                    self.index[name] = meta
                }
                lock.unlock()
            } catch {
                print("[WeightLoader] Failed to parse header of \(url.lastPathComponent): \(error)")
            }
        }
        
        // Identify global (non-layer) tensors and pre-load them.
        // "Global" = always-resident. Includes the embed/lm_head/top-
        // level norm/hc_head params that every token needs.
        //
        // EXPLICITLY EXCLUDE:
        //   - `layers.*` — those are streamed per-layer in forward.
        //   - `mtp.*`    — MTP layers carry a full transformer block
        //                  (256 routed experts, MLA, etc. ≈ 6 GB per
        //                  block on V4-Pro). They are NOT invoked by
        //                  the current `Transformer.forward` (mtp:
        //                  [] passed in Assembly.load), so preloading
        //                  them just pins 6 GB resident forever and
        //                  was a major contributor to the >25 GB
        //                  peak the user observed.
        //   - `compressor.*`/`indexer.*` at top level — same reason,
        //                  not wired in the current MLX forward.
        for name in index.keys {
            if name.hasPrefix("layers.") { continue }
            if name.hasPrefix("mtp.") { continue }
            globalNames.insert(name)
        }

        // Pre-load global tensors (embed, head, norm, hc_head_*).
        preloadTensors(names: globalNames)
    }
    
    // MARK: - Header parsing
    
    /// Parses only the JSON header of a safetensors file. Returns
    /// tensor name → TensorMeta without reading any tensor data.
    private static func parseHeader(url: URL) throws -> [String: TensorMeta] {
        let fh = try FileHandle(forReadingFrom: url)
        defer { try? fh.close() }
        
        // First 8 bytes: little-endian uint64 header size
        guard let sizeData = try fh.read(upToCount: 8), sizeData.count == 8 else {
            throw NSError(domain: "WeightLoader", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Failed to read header size from \(url.lastPathComponent)"
            ])
        }
        let headerSize = sizeData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        
        // Read the JSON header
        guard let jsonData = try fh.read(upToCount: Int(headerSize)) else {
            throw NSError(domain: "WeightLoader", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Failed to read header JSON from \(url.lastPathComponent)"
            ])
        }
        
        guard let json = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw NSError(domain: "WeightLoader", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Invalid header JSON in \(url.lastPathComponent)"
            ])
        }
        
        let dataStartOffset = 8 + Int(headerSize)
        var result: [String: TensorMeta] = [:]
        
        for (key, value) in json {
            // Skip the __metadata__ key
            if key == "__metadata__" { continue }
            
            guard let info = value as? [String: Any],
                  let dtypeStr = info["dtype"] as? String,
                  let shape = info["shape"] as? [Int],
                  let offsets = info["data_offsets"] as? [Int],
                  offsets.count == 2
            else { continue }
            
            result[key] = TensorMeta(
                dtype: dtypeStr,
                shape: shape,
                offsetBegin: offsets[0],
                offsetEnd: offsets[1],
                shardURL: url,
                dataOffset: dataStartOffset
            )
        }
        
        return result
    }
    
    // MARK: - On-demand tensor loading
    
    /// Load a single tensor from disk by reading only its byte range.
    private func loadTensorFromDisk(name: String, meta: TensorMeta) -> MLXArray? {
        do {
            let fh = try FileHandle(forReadingFrom: meta.shardURL)
            defer { try? fh.close() }
            
            let absOffset = UInt64(meta.dataOffset + meta.offsetBegin)
            let byteCount = meta.offsetEnd - meta.offsetBegin
            
            try fh.seek(toOffset: absOffset)
            guard let data = try fh.read(upToCount: byteCount), data.count == byteCount else {
                return nil
            }
            
            let (_, mlxDType) = dtypeFromString(meta.dtype)
            return MLXArray(data, meta.shape, dtype: mlxDType)
        } catch {
            print("[WeightLoader] Error loading tensor '\(name)': \(error)")
            return nil
        }
    }
    
    /// Pre-load a set of tensor names into the cache (used for globals).
    private func preloadTensors(names: Set<String>) {
        let namesToLoad = names.filter { index[$0] != nil }
        let nameArray = Array(namesToLoad)
        
        DispatchQueue.concurrentPerform(iterations: nameArray.count) { i in
            let name = nameArray[i]
            guard let meta = self.index[name] else { return }
            guard let arr = self.loadTensorFromDisk(name: name, meta: meta) else { return }
            self.cacheLock.lock()
            self.cache[name] = arr
            self.cacheLock.unlock()
        }
    }
    
    // MARK: - Layer streaming API

    /// Returns true if `name` belongs to a routed (non-shared) expert
    /// under any layer — i.e. `layers.<K>.ffn.experts.<E>.*`. These are
    /// the tensors we defer to MoE-time loading so we only ever hold
    /// `topK` (≈ 8) of `nRoutedExperts` (≈ 256) per layer resident.
    private static func isRoutedExpertTensor(_ name: String) -> Bool {
        // Cheap structural test: split on '.' and look for the segment
        // pattern `ffn.experts.<int>`. The shared expert
        // (`ffn.shared_experts.*`) is NOT routed and stays resident.
        let parts = name.split(separator: ".")
        guard parts.count >= 4 else { return false }
        for i in 0..<(parts.count - 2) {
            if parts[i] == "ffn", parts[i + 1] == "experts",
               Int(parts[i + 2]) != nil {
                return true
            }
        }
        return false
    }

    private func ensureNames(_ toLoad: [(String, TensorMeta)]) {
        guard !toLoad.isEmpty else { return }
        // Bound concurrent disk reads. `DispatchQueue.concurrentPerform`
        // fans out to `iterations` workers in parallel — for a bulk
        // load of 256 routed experts × 6 tensors (≈ 1500 items) every
        // outstanding read holds a `Data` buffer in RAM until the
        // MLXArray copy completes, so the in-flight buffer total can
        // approach the sum of every tensor still to load. Cap the
        // fan-out so the transient peak is bounded by
        // `maxConcurrent × avg_tensor_size`. Override via
        // DEEPSEEK_LOAD_CONCURRENCY.
        let maxConcurrent: Int = {
            if let raw = ProcessInfo.processInfo.environment["DEEPSEEK_LOAD_CONCURRENCY"],
               let n = Int(raw), n >= 1 { return n }
            return 8
        }()

        let semaphore = DispatchSemaphore(value: maxConcurrent)
        let group = DispatchGroup()
        let queue = DispatchQueue.global(qos: .userInitiated)

        for (name, meta) in toLoad {
            semaphore.wait()
            group.enter()
            queue.async { [weak self] in
                defer {
                    semaphore.signal()
                    group.leave()
                }
                guard let self = self else { return }
                guard let arr = self.loadTensorFromDisk(name: name, meta: meta) else { return }
                self.cacheLock.lock()
                self.cache[name] = arr
                self.cacheLock.unlock()
            }
        }
        group.wait()
    }

    /// Load layer `K`'s non-routed-expert tensors into cache. Called by
    /// `Transformer.forward()` before processing layer K. Routed
    /// experts are loaded on demand by `MoEFFN` once the gate has run
    /// and we know which `topK` ids are active — that cuts the
    /// resident working set of a MoE layer from `nRoutedExperts × FFN`
    /// to `topK × FFN`.
    public func ensureLayer(_ K: Int) {
        let prefix = "layers.\(K)."
        var toLoad: [(String, TensorMeta)] = []

        for (name, meta) in index {
            guard name.hasPrefix(prefix) else { continue }
            if Self.isRoutedExpertTensor(name) { continue }
            cacheLock.lock()
            let alreadyCached = cache[name] != nil
            cacheLock.unlock()
            if !alreadyCached {
                toLoad.append((name, meta))
            }
        }
        ensureNames(toLoad)
    }

    /// Load the given routed experts of layer `K` into cache. Idempotent.
    public func ensureExperts(layer K: Int, indices: [Int]) {
        var toLoad: [(String, TensorMeta)] = []

        for e in indices {
            let prefix = "layers.\(K).ffn.experts.\(e)."
            for (name, meta) in index {
                if name.hasPrefix(prefix) {
                    cacheLock.lock()
                    let alreadyCached = cache[name] != nil
                    cacheLock.unlock()
                    if !alreadyCached {
                        toLoad.append((name, meta))
                    }
                }
            }
        }
        ensureNames(toLoad)
    }

    /// Drop the given routed experts of layer `K` from cache. Called
    /// by `MoEFFN` after the dispatch loop so the next layer doesn't
    /// inherit the residents.
    public func releaseExperts(layer K: Int, indices: [Int]) {
        cacheLock.lock()
        // Collect first, mutate after — iterating `cache.keys` while
        // calling `removeValue` is undefined behavior in Swift and can
        // silently skip entries (which would leak expert weights
        // across layers).
        var keysToRemove: [String] = []
        for e in indices {
            let prefix = "layers.\(K).ffn.experts.\(e)."
            for name in cache.keys where name.hasPrefix(prefix) {
                keysToRemove.append(name)
            }
        }
        for name in keysToRemove {
            cache.removeValue(forKey: name)
        }
        cacheLock.unlock()
    }

    /// Start loading layer K+1's non-expert tensors in background
    /// while layer K executes. The routed experts of layer K+1 are
    /// fetched after its own gate runs, so they're not prefetched.
    public func prefetchLayer(_ layerIndex: Int) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.ensureLayer(layerIndex)
        }
    }

    /// Release all tensors for layer `K` from cache, freeing RAM.
    /// Removes both shared and routed-expert entries (the latter are
    /// usually already released by MoEFFN, but this is idempotent).
    public func releaseLayer(_ layerIndex: Int) {
        let prefix = "layers.\(layerIndex)."
        cacheLock.lock()
        var keysToRemove: [String] = []
        for name in cache.keys where name.hasPrefix(prefix) {
            keysToRemove.append(name)
        }
        for name in keysToRemove {
            cache.removeValue(forKey: name)
        }
        cacheLock.unlock()
    }

    /// Returns the number of tensors currently held in the on-demand
    /// cache (globals are counted too). Useful to verify that the
    /// streaming releases are actually freeing entries between layers.
    public var cachedTensorCount: Int {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return cache.count
    }
    
    // MARK: - Tensor access API (used during Assembly)
    
    public func load(_ name: String) throws -> Tensor? {
        // Check cache first
        cacheLock.lock()
        if let arr = cache[name] {
            cacheLock.unlock()
            let (dt, _) = index[name].map { dtypeFromString($0.dtype) } ?? (.f32, .float32)
            return Tensor(array: arr, dtype: dt)
        }
        cacheLock.unlock()
        
        // Not cached — try to load from disk on demand
        if let meta = index[name] {
            if let arr = loadTensorFromDisk(name: name, meta: meta) {
                let (dt, _) = dtypeFromString(meta.dtype)
                // Cache it if it's a global tensor
                if globalNames.contains(name) {
                    cacheLock.lock()
                    cache[name] = arr
                    cacheLock.unlock()
                }
                return Tensor(array: arr, dtype: dt)
            }
        }
        
        missingLock.lock()
        missing.insert(name)
        missingLock.unlock()
        return nil
    }
    
    public func dtype(of name: String) -> DType? {
        if let dtypeStr = index[name]?.dtype {
            let (dt, _) = dtypeFromString(dtypeStr)
            return dt
        }
        return nil
    }

    public func dtype(ofAny candidates: [String]) -> DType? {
        for n in candidates {
            if let dt = dtype(of: n) { return dt }
        }
        return nil
    }

    public func shape(of name: String) -> [Int]? {
        return index[name]?.shape
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
        missingLock.lock()
        for n in candidates { missing.insert(n) }
        missingLock.unlock()
        return nil
    }
    
    @discardableResult
    public func warmupAllShards(memoryGuardRatio: Double = 1.5) -> Bool {
        // In streaming mode, warmup is a no-op — we load on demand.
        return true
    }
}
