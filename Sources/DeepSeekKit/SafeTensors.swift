import Foundation
import Metal

/// Minimal safetensors reader. The format is:
///   [u64 little-endian header length][JSON header][tensor data...]
/// Header maps tensor name -> { dtype, shape, data_offsets:[start,end] }.
///
/// DeepSeek-V4 ships weights in a sharded form indexed by
/// `model.safetensors.index.json`; the multi-rank Python loader uses
/// `model{rank}-mp{world_size}.safetensors` after `convert.py` does the
/// MP repack. For single-rank macOS we want world_size = 1 → use
/// `convert.py --n-experts ... --model-parallel 1` upstream.
public final class SafeTensorsFile {
    public struct Entry: Decodable {
        public let dtype: String
        public let shape: [Int]
        public let dataOffsets: [Int]
        enum CodingKeys: String, CodingKey {
            case dtype, shape
            case dataOffsets = "data_offsets"
        }
    }

    public let url: URL
    public let entries: [String: Entry]
    private let dataStart: Int
    private let fileHandle: FileHandle

    public init(url: URL) throws {
        self.url = url
        let fh = try FileHandle(forReadingFrom: url)
        self.fileHandle = fh

        try fh.seek(toOffset: 0)
        guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else {
            throw NSError(domain: "SafeTensors", code: 1)
        }
        let headerLen = lenData.withUnsafeBytes { $0.load(as: UInt64.self) }
        guard let header = try fh.read(upToCount: Int(headerLen)) else {
            throw NSError(domain: "SafeTensors", code: 2)
        }

        let raw = try JSONSerialization.jsonObject(with: header) as? [String: Any] ?? [:]
        var parsed: [String: Entry] = [:]
        for (k, v) in raw {
            if k == "__metadata__" { continue }
            let entryData = try JSONSerialization.data(withJSONObject: v)
            parsed[k] = try JSONDecoder().decode(Entry.self, from: entryData)
        }
        self.entries = parsed
        self.dataStart = 8 + Int(headerLen)
    }

    public func load(_ name: String, on device: Device = .shared) throws -> Tensor {
        guard let e = entries[name] else {
            throw NSError(domain: "SafeTensors", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "missing tensor \(name)"])
        }
        let start = dataStart + e.dataOffsets[0]
        let end = dataStart + e.dataOffsets[1]
        let len = end - start
        try fileHandle.seek(toOffset: UInt64(start))
        guard let bytes = try fileHandle.read(upToCount: len), bytes.count == len else {
            throw NSError(domain: "SafeTensors", code: 4)
        }
        let dt = Self.parseDType(e.dtype)
        return bytes.withUnsafeBytes { raw in
            Tensor.from(bytes: raw, shape: e.shape, dtype: dt, on: device)
        }
    }

    private static func parseDType(_ s: String) -> DType {
        switch s.uppercased() {
        case "F32": return .f32
        case "F16": return .f16
        case "BF16": return .bf16
        case "I32": return .i32
        case "I8", "U8": return .i8
        // PyTorch float8/float4 dtypes serialized by safetensors:
        case "F8_E4M3", "F8E4M3", "FLOAT8_E4M3FN": return .fp8E4M3
        case "F4_E2M1", "F4E2M1", "FLOAT4_E2M1FN_X2": return .fp4E2M1
        case "F8_E8M0", "F8E8M0", "FLOAT8_E8M0FNU": return .e8m0
        default:
            fatalError("unsupported safetensors dtype: \(s)")
        }
    }
}
