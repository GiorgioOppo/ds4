import Foundation
import DS4Core

// MARK: - Frames

/// Fixed header preceding every framed message: magic + type + payload length.
public struct DistFrameHeader {
    public static let byteSize = 12
    public var type: Dist.MsgType
    public var length: UInt32     // payload bytes following the header

    public func encoded() -> Data {
        var d = Data(capacity: DistFrameHeader.byteSize)
        d.appendLE(Dist.magic)
        d.appendLE(type.rawValue)
        d.appendLE(length)
        return d
    }

    public static func decode(_ d: Data) -> DistFrameHeader? {
        guard d.count >= byteSize else { return nil }
        var o = d.startIndex
        guard d.readLE(&o) == Dist.magic,
              let type = Dist.MsgType(rawValue: d.readLE(&o)) else { return nil }
        let length = d.readLE(&o) as UInt32
        return DistFrameHeader(type: type, length: length)
    }
}

