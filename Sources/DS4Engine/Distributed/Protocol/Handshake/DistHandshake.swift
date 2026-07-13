import Foundation
import DS4Core

/// HELLO payload: a worker announces its protocol version, its state
/// (`assigned` — an engine is loaded for a slice) and, when assigned, the model
/// identity and slice it serves. Sent on connect and echoed as the READY
/// payload after an ASSIGN completes. The coordinator validates the version
/// FIRST — a mixed cluster fails with a clear error instead of garbled frames.
public struct DistHello: Sendable {
    public var version: UInt32
    public var assigned: Bool         // an engine is loaded (slice fields valid)
    public var modelName: String      // loaded gguf (assigned) or local hint (idle)
    public var layerStart: Int
    public var layerEnd: Int          // inclusive
    public var hasOutput: Bool        // also owns the output head
    public var nLayers: Int
    public var contextSize: Int       // 0 while idle

    public init(modelName: String, layerStart: Int, layerEnd: Int, hasOutput: Bool,
                nLayers: Int, contextSize: Int, assigned: Bool = true,
                version: UInt32 = Dist.protocolVersion) {
        self.version = version
        self.assigned = assigned
        self.modelName = modelName; self.layerStart = layerStart; self.layerEnd = layerEnd
        self.hasOutput = hasOutput; self.nLayers = nLayers; self.contextSize = contextSize
    }

    /// An unconfigured worker waiting for an ASSIGN.
    public static func idle(localModelName: String, nLayers: Int) -> DistHello {
        DistHello(modelName: localModelName, layerStart: 0, layerEnd: 0, hasOutput: false,
                  nLayers: nLayers, contextSize: 0, assigned: false)
    }

    public func encoded() -> Data {
        var d = Data()
        d.appendLE(version)
        d.appendLE(UInt32(assigned ? 1 : 0))
        d.appendLE(UInt32(layerStart)); d.appendLE(UInt32(layerEnd))
        d.appendLE(UInt32(hasOutput ? 1 : 0))
        d.appendLE(UInt32(nLayers)); d.appendLE(UInt32(contextSize))
        let name = Data(modelName.utf8)
        d.appendLE(UInt32(name.count)); d.append(name)
        return d
    }

    public static func decode(_ d: Data) -> DistHello? {
        var o = d.startIndex
        guard d.count >= 32 else { return nil }
        let ver = d.readLE(&o) as UInt32
        let assigned = (d.readLE(&o) as UInt32) != 0
        let ls = Int(d.readLE(&o) as UInt32), le = Int(d.readLE(&o) as UInt32)
        let ho = (d.readLE(&o) as UInt32) != 0
        let nl = Int(d.readLE(&o) as UInt32), ctx = Int(d.readLE(&o) as UInt32)
        let nameLen = Int(d.readLE(&o) as UInt32)
        guard nameLen >= 0, o + nameLen <= d.endIndex else { return nil }
        let name = String(decoding: d[o..<o+nameLen], as: UTF8.self)
        return DistHello(modelName: name, layerStart: ls, layerEnd: le, hasOutput: ho,
                         nLayers: nl, contextSize: ctx, assigned: assigned, version: ver)
    }
}

/// ASSIGN payload: the coordinator defines a worker's whole job — WHICH gguf
/// (full coordinator-side path + filename, resolved locally by the worker),
/// the context size, the expert-cache budget, the disk-KV budget, the usage
/// imatrix to pre-warm the slot cache with, and the layer slice to own.
/// The worker loads (or reuses) its engine and replies READY (HELLO payload).
public struct DistAssign: Sendable {
    public var modelPath: String      // coordinator's path (verbatim, tried first)
    public var modelName: String      // gguf filename (fallback resolution key)
    public var contextSize: Int
    public var expertCacheSlots: Int  // 0 = no expert slot-cache on the worker
    public var diskKVBudgetTokens: Int // 0 = no disk-KV checkpoints on the worker
    public var useExpertBundle: Bool   // worker runs with the expert-bundle sidecar
    public var useDenseQ4: Bool        // worker uses the Q4 dense requant (cache offered)
    public var layerStart: Int
    public var layerEnd: Int          // inclusive
    public var hasOutput: Bool        // last slice: also runs the output head
    /// Usage-imatrix JSON (ExpertUsageStats.serialize) to seed the worker's
    /// slot-cache pre-warm; empty = none (the worker may still have its own).
    public var usageJSON: Data
    /// v9: the coordinator's PERFORMANCE env (whitelisted, Dist.perfKnobKeys).
    /// The worker applies these before loading its engine, so a shard runs
    /// with the coordinator's measured configuration instead of whatever the
    /// worker app's local defaults happen to be.
    public var envKnobs: [(key: String, value: String)]

    public init(modelPath: String, modelName: String, contextSize: Int, expertCacheSlots: Int,
                diskKVBudgetTokens: Int, useExpertBundle: Bool = false, useDenseQ4: Bool = false,
                layerStart: Int, layerEnd: Int, hasOutput: Bool,
                usageJSON: Data = Data(), envKnobs: [(key: String, value: String)] = []) {
        self.modelPath = modelPath; self.modelName = modelName
        self.contextSize = contextSize; self.expertCacheSlots = expertCacheSlots
        self.diskKVBudgetTokens = diskKVBudgetTokens
        self.useExpertBundle = useExpertBundle
        self.useDenseQ4 = useDenseQ4
        self.layerStart = layerStart; self.layerEnd = layerEnd; self.hasOutput = hasOutput
        self.usageJSON = usageJSON
        self.envKnobs = envKnobs
    }

    public func encoded() -> Data {
        var d = Data()
        d.appendLE(UInt32(contextSize))
        d.appendLE(UInt32(expertCacheSlots))
        d.appendLE(UInt32(diskKVBudgetTokens))
        d.appendLE(UInt32(useExpertBundle ? 1 : 0))
        d.appendLE(UInt32(useDenseQ4 ? 1 : 0))
        d.appendLE(UInt32(layerStart)); d.appendLE(UInt32(layerEnd))
        d.appendLE(UInt32(hasOutput ? 1 : 0))
        let path = Data(modelPath.utf8)
        d.appendLE(UInt32(path.count)); d.append(path)
        let name = Data(modelName.utf8)
        d.appendLE(UInt32(name.count)); d.append(name)
        d.appendLE(UInt32(usageJSON.count)); d.append(usageJSON)
        d.appendLE(UInt32(envKnobs.count))
        for (k, v) in envKnobs {
            let kd = Data(k.utf8), vd = Data(v.utf8)
            d.appendLE(UInt32(kd.count)); d.append(kd)
            d.appendLE(UInt32(vd.count)); d.append(vd)
        }
        return d
    }

    public static func decode(_ d: Data) -> DistAssign? {
        var o = d.startIndex
        guard d.count >= 48 else { return nil }
        let ctx = Int(d.readLE(&o) as UInt32)
        let slots = Int(d.readLE(&o) as UInt32)
        let kvBudget = Int(d.readLE(&o) as UInt32)
        let bundle = (d.readLE(&o) as UInt32) != 0
        let q4 = (d.readLE(&o) as UInt32) != 0
        let ls = Int(d.readLE(&o) as UInt32), le = Int(d.readLE(&o) as UInt32)
        let ho = (d.readLE(&o) as UInt32) != 0
        let pathLen = Int(d.readLE(&o) as UInt32)
        guard pathLen >= 0, o + pathLen + 4 <= d.endIndex else { return nil }
        let path = String(decoding: d[o..<o+pathLen], as: UTF8.self); o += pathLen
        let nameLen = Int(d.readLE(&o) as UInt32)
        guard nameLen >= 0, o + nameLen + 4 <= d.endIndex else { return nil }
        let name = String(decoding: d[o..<o+nameLen], as: UTF8.self); o += nameLen
        let usageLen = Int(d.readLE(&o) as UInt32)
        guard usageLen >= 0, o + usageLen + 4 <= d.endIndex else { return nil }
        let usage = Data(d[o..<o+usageLen]); o += usageLen
        let knobCount = Int(d.readLE(&o) as UInt32)
        guard knobCount >= 0, knobCount <= 64 else { return nil }
        var knobs: [(key: String, value: String)] = []
        knobs.reserveCapacity(knobCount)
        for _ in 0..<knobCount {
            guard o + 4 <= d.endIndex else { return nil }
            let kLen = Int(d.readLE(&o) as UInt32)
            guard kLen > 0, kLen <= 256, o + kLen + 4 <= d.endIndex else { return nil }
            let k = String(decoding: d[o..<o+kLen], as: UTF8.self); o += kLen
            let vLen = Int(d.readLE(&o) as UInt32)
            guard vLen >= 0, vLen <= 256, o + vLen <= d.endIndex else { return nil }
            let v = String(decoding: d[o..<o+vLen], as: UTF8.self); o += vLen
            knobs.append((key: k, value: v))
        }
        return DistAssign(modelPath: path, modelName: name, contextSize: ctx,
                          expertCacheSlots: slots, diskKVBudgetTokens: kvBudget,
                          useExpertBundle: bundle, useDenseQ4: q4,
                          layerStart: ls, layerEnd: le, hasOutput: ho, usageJSON: usage,
                          envKnobs: knobs)
    }
}

