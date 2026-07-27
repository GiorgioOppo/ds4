import Foundation

// Read-back helpers that turn a loaded GGUFModel into the value/tensor inputs the
// GGUFWriter consumes. This closes the round-trip (read -> edit -> write) that an
// offline requantizer needs: metadata is re-materialized as typed
// GGUFMetadataValue, tensor bytes are copied out of the mmap by absolute offset.

extension GGUFModel {

    /// All metadata KVs in file order, decoded to typed values. Order is
    /// preserved so a re-written file keeps the original layout.
    public func allMetadata() throws -> [(key: String, value: GGUFMetadataValue)] {
        var out: [(String, GGUFMetadataValue)] = []
        out.reserveCapacity(kvs.count)
        let b = mapBase.assumingMemoryBound(to: UInt8.self)
        for kv in kvs {
            var c = GGUFCursor(base: b, size: size, pos: kv.valuePos)
            out.append((kv.key, try Self.readValue(&c, type: kv.type)))
        }
        return out
    }

    /// A copy of a tensor's raw block bytes, straight from the mapping.
    public func tensorData(_ t: Tensor) -> Data {
        Data(bytes: mapBase.advanced(by: Int(t.absOffset)), count: Int(t.bytes))
    }

    /// Convenience: build a `GGUFWriter.TensorInput` that reuses this tensor's
    /// bytes unchanged (for tensors an offline pass copies through verbatim).
    public func tensorInputPassthrough(_ t: Tensor) -> GGUFWriter.TensorInput {
        GGUFWriter.TensorInput(name: t.name, dims: t.dims, type: t.type, data: tensorData(t))
    }

    // MARK: - Value decoding (payload only; the type is already known)

    static func readValue(_ c: inout GGUFCursor, type: UInt32) throws -> GGUFMetadataValue {
        switch type {
        case GGUFValueType.uint8.rawValue:   return .uint8(try c.read(as: UInt8.self))
        case GGUFValueType.int8.rawValue:    return .int8(try c.read(as: Int8.self))
        case GGUFValueType.uint16.rawValue:  return .uint16(UInt16(littleEndian: try c.read(as: UInt16.self)))
        case GGUFValueType.int16.rawValue:   return .int16(Int16(bitPattern: UInt16(littleEndian: try c.read(as: UInt16.self))))
        case GGUFValueType.uint32.rawValue:  return .uint32(try c.u32())
        case GGUFValueType.int32.rawValue:   return .int32(Int32(bitPattern: try c.u32()))
        case GGUFValueType.uint64.rawValue:  return .uint64(try c.u64())
        case GGUFValueType.int64.rawValue:   return .int64(Int64(bitPattern: try c.u64()))
        case GGUFValueType.float32.rawValue: return .float32(Float(bitPattern: try c.u32()))
        case GGUFValueType.float64.rawValue: return .float64(Double(bitPattern: try c.u64()))
        case GGUFValueType.bool.rawValue:    return .bool(try c.read(as: UInt8.self) != 0)
        case GGUFValueType.string.rawValue:  return .string(try c.stringBytes())
        case GGUFValueType.array.rawValue:
            let itemType = try c.u32()
            let len = try c.u64()
            guard let et = GGUFValueType(rawValue: itemType) else {
                throw GGUFError.message("GGUF array has unknown element type \(itemType)")
            }
            var elements: [GGUFMetadataValue] = []
            elements.reserveCapacity(Int(len))
            for _ in 0..<len { elements.append(try readValue(&c, type: itemType)) }
            return .array(elementType: et, elements: elements)
        default:
            throw GGUFError.message("cannot decode GGUF metadata value of type \(type)")
        }
    }
}
