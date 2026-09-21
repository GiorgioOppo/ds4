import Foundation
import Darwin
import DS4Core

struct DeepSeek41Weights {
    let resident: [String: DS41Weight]
    let experts: [[String: GGUFModel.Tensor]]
    let maximumGateBytes: Int, maximumDownBytes: Int

    init(model: GGUFModel, runtime: MetalRuntime) throws {
        var specs: [String: GGUFModel.Tensor] = [:]
        var experts: [[String: GGUFModel.Tensor]] = []
        var maxGate = 0, maxDown = 0
        func check(_ name: String, _ dims: [Int], _ types: [UInt32], resident: Bool = true) throws -> GGUFModel.Tensor {
            guard let tensor = model.findTensor(name), tensor.dims == dims.map(UInt64.init), types.contains(tensor.type),
                  let bytes = GGUF.tensorNBytes(type:tensor.type,elements:tensor.elements), bytes == tensor.bytes,
                  bytes > 0, tensor.absOffset <= model.size, bytes <= model.size - tensor.absOffset,
                  tensor.absOffset % (tensor.type == 0 ? 4 : 2) == 0 else { throw DeepSeek41Error.invalidTensor(name) }
            if resident { specs[name] = tensor }
            return tensor
        }
        let dense: [UInt32] = [8,12,2], routed: [UInt32] = [8,10,12,13,14,16,39]
        _ = try check("token_embd.weight",[5120,129280],[1])
        _ = try check("output_norm.weight",[5120],[0])
        _ = try check("output.weight",[5120,129280],dense)
        for i in 0..<40 {
            let p = "blk.\(i)."
            for kind in ["attn","ffn"] {
                _ = try check(p+"hc_"+kind+"_fn.weight",[20480,24],[1])
                _ = try check(p+"hc_"+kind+"_scale.weight",[3],[0])
                _ = try check(p+"hc_"+kind+"_base.weight",[24],[0])
                _ = try check(p+kind+"_norm.weight",[5120],[0])
            }
            _ = try check(p+"attn_q_a.weight",[5120,1280],dense)
            _ = try check(p+"attn_q_a_norm.weight",[1280],[0])
            _ = try check(p+"attn_q_b.weight",[1280,32768],dense)
            _ = try check(p+"attn_kv.weight",[5120,512],dense)
            _ = try check(p+"attn_kv_a_norm.weight",[512],[0])
            _ = try check(p+"attn_sinks.weight",[64],[0])
            _ = try check(p+"attn_output_a.weight",[4096,8192],[8])
            _ = try check(p+"attn_output_b.weight",[8192,5120],[8])
            _ = try check(p+"ffn_gate_inp.weight",[5120,384],[0])
            _ = try check(p+"exp_probs_b.bias",[384],[0])
            if model.findTensor(p+"exp_probs_b_vl.bias") != nil { _ = try check(p+"exp_probs_b_vl.bias",[384],[0]) }
            for kind in ["gate","up"] { _ = try check(p+"ffn_"+kind+"_shexp.weight",[5120,2304],dense) }
            _ = try check(p+"ffn_down_shexp.weight",[2304,5120],dense)
            var record: [String: GGUFModel.Tensor] = [:]
            for kind in ["gate","up","down"] {
                let tensor = try check(p+"ffn_"+kind+"_exps.weight",kind == "down" ? [2304,5120,384] : [5120,2304,384],routed,resident:false)
                record[kind] = tensor
                if kind == "down" { maxDown = max(maxDown,Int(tensor.bytes/384)) }
                else { maxGate = max(maxGate,Int(tensor.bytes/384)) }
            }
            guard record["gate"]!.type == record["up"]!.type else { throw DeepSeek41Error.invalidTensor(p+"ffn_up_exps.weight") }
            experts.append(record)
            if DeepSeek41Configuration.kvSourceLayers.contains(i) {
                _ = try check(p+"attn_compressor_kv.weight",[5120,512],[1])
                _ = try check(p+"attn_compressor_norm.weight",[512],[0])
                if i < 20 { _ = try check(p+"attn_compressor_gate.weight",[5120,512],[1]) }
                _ = try check(p+"indexer.attn_k.weight",[512,128],[1])
                _ = try check(p+"indexer.k_norm.weight",[128],[0])
            }
            if DeepSeek41Configuration.indexSourceLayers.contains(i) {
                _ = try check(p+"indexer.attn_q_b.weight",[1280,4096],[1])
                _ = try check(p+"indexer.proj.weight",[5120,32],[1])
            }
            if [1,14].contains(i) {
                _ = try check(p+"engram_kv.weight",[6144,25600],[1])
                _ = try check(p+"engram_q_norm.weight",[5120,4],[0])
                _ = try check(p+"engram_k_norm.weight",[5120,4],[0])
            }
        }
        let bytes = specs.values.reduce(UInt64(0)) { $0 + $1.bytes + UInt64(2*getpagesize()) }
        guard bytes < runtime.device.recommendedMaxWorkingSetSize else {
            throw SwiftModelDecoderError.gpu("V4.1: pesi dense oltre la memoria Metal disponibile")
        }
        var resident: [String: DS41Weight] = [:]
        for (name,tensor) in specs {
            let length = Int(tensor.bytes), pointer = model.mapBase.advanced(by:Int(tensor.absOffset))
            guard length + 2*Int(getpagesize()) <= runtime.device.maxBufferLength else { throw DeepSeek41Error.invalidTensor(name) }
            let gpu = try length < 65536
                ? GPUTensor.raw(runtime,ptr:pointer,byteLength:length,elementCount:Int(tensor.elements))
                : GPUTensor.mappedNoCopy(runtime,ptr:pointer,byteLength:length,elementCount:Int(tensor.elements))
            resident[name] = DS41Weight(tensor:tensor,gpu:gpu,columns:Int(tensor.dims[0]),rows:tensor.dims.count > 1 ? Int(tensor.dims[1]) : 1)
        }
        self.resident = resident; self.experts = experts; maximumGateBytes = maxGate; maximumDownBytes = maxDown
    }
}

/// Six records are a bounded wave. All selected rows using a record execute
/// together; a record is overwritten only after that wave has completed.
final class DeepSeek41ExpertPager {
    private let model: GGUFModel
    private let weights: DeepSeek41Weights
    // Borrowed descriptor: GGUFModel owns/caches uncachedFD and closes it in
    // deinit. Retaining model keeps it alive; closing here would double-close.
    private let fd: Int32
    private let gate: [GPUTensor], up: [GPUTensor], down: [GPUTensor]
    init(model: GGUFModel, weights: DeepSeek41Weights, runtime: MetalRuntime) throws {
        guard let fd = model.uncachedFD() else { throw DeepSeek41Error.invalidTensor("cannot open expert SSD reader") }
        self.model = model; self.weights = weights; self.fd = fd
        gate = try (0..<6).map { _ in try .uninitializedBytes(runtime,byteLength:weights.maximumGateBytes,elementCount:5120*2304) }
        up = try (0..<6).map { _ in try .uninitializedBytes(runtime,byteLength:weights.maximumGateBytes,elementCount:5120*2304) }
        down = try (0..<6).map { _ in try .uninitializedBytes(runtime,byteLength:weights.maximumDownBytes,elementCount:5120*2304) }
    }
    func read(layer: Int, expert: Int, slot: Int, cancelled: @Sendable () -> Bool) throws -> (DS41Weight,DS41Weight,DS41Weight) {
        guard (0..<384).contains(expert), (0..<6).contains(slot) else { throw DeepSeek41Error.invalidTensor("expert ID") }
        var result: [DS41Weight] = []
        for (kind,buffer) in [("gate",gate[slot]),("up",up[slot]),("down",down[slot])] {
            let tensor = weights.experts[layer][kind]!, length = Int(tensor.bytes/384)
            let offset = tensor.absOffset + UInt64(expert*length)
            var readBytes = 0
            while readBytes < length {
                if cancelled() { throw SwiftModelDecoderError.cancelled }
                let amount = min(length-readBytes,4*1024*1024)
                let n = pread(fd,buffer.buffer.contents().advanced(by:readBytes),amount,off_t(offset+UInt64(readBytes)))
                if n < 0 && errno == EINTR { continue }
                guard n > 0 else { throw DeepSeek41Error.invalidTensor("short read \(tensor.name) expert \(expert)") }
                readBytes += n
            }
            result.append(DS41Weight(tensor:tensor,gpu:buffer,columns:Int(tensor.dims[0]),rows:Int(tensor.dims[1])))
        }
        return (result[0],result[1],result[2])
    }
}
