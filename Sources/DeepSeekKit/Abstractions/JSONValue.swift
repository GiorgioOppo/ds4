import Foundation

/// Tipizzazione strict-Sendable di un grafo JSON. Stessa identica
/// definizione di
/// `Sources/DeepSeekTools/Abstractions/JSONValue.swift` —
/// duplicazione deliberata per evitare di imporre una dipendenza
/// fra `DeepSeekKit` e `DeepSeekTools` (vedi `Package.swift:54-61`,
/// `Has no Metal / DeepSeekKit dependency on purpose`).
///
/// DeepSeekUI importa entrambi i target e può convertire fra le
/// due forme passando per `foundationValue` / `init(any:)`.
public indirect enum JSONValue: Sendable, Codable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    // MARK: - Codable

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
            return
        }
        if let v = try? c.decode(Bool.self) {
            self = .bool(v)
            return
        }
        if let v = try? c.decode(Int.self) {
            self = .int(v)
            return
        }
        if let v = try? c.decode(Double.self) {
            self = .double(v)
            return
        }
        if let v = try? c.decode(String.self) {
            self = .string(v)
            return
        }
        if let v = try? c.decode([JSONValue].self) {
            self = .array(v)
            return
        }
        if let v = try? c.decode([String: JSONValue].self) {
            self = .object(v)
            return
        }
        throw DecodingError.dataCorruptedError(
            in: c,
            debugDescription: "Unsupported JSON value")
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null:           try c.encodeNil()
        case .bool(let v):    try c.encode(v)
        case .int(let v):     try c.encode(v)
        case .double(let v):  try c.encode(v)
        case .string(let v):  try c.encode(v)
        case .array(let v):   try c.encode(v)
        case .object(let v):  try c.encode(v)
        }
    }

    // MARK: - Foundation interop

    public var foundationValue: Any {
        switch self {
        case .null:           return NSNull()
        case .bool(let v):    return v
        case .int(let v):     return v
        case .double(let v):  return v
        case .string(let v):  return v
        case .array(let v):   return v.map { $0.foundationValue }
        case .object(let v):  return v.mapValues { $0.foundationValue }
        }
    }

    public init(any: Any) {
        if any is NSNull {
            self = .null
            return
        }
        if let n = any as? NSNumber {
            let t = String(cString: n.objCType)
            if t == "c" || t == "B" {
                self = .bool(n.boolValue)
                return
            }
            if t == "f" || t == "d" {
                self = .double(n.doubleValue)
                return
            }
            self = .int(n.intValue)
            return
        }
        if let v = any as? Bool {
            self = .bool(v)
            return
        }
        if let v = any as? Int {
            self = .int(v)
            return
        }
        if let v = any as? Double {
            self = .double(v)
            return
        }
        if let v = any as? String {
            self = .string(v)
            return
        }
        if let v = any as? [Any] {
            self = .array(v.map { JSONValue(any: $0) })
            return
        }
        if let v = any as? [String: Any] {
            self = .object(v.mapValues { JSONValue(any: $0) })
            return
        }
        self = .null
    }

    // MARK: - Convenience accessors

    public var asObject: [String: JSONValue]? {
        if case .object(let o) = self { return o }
        return nil
    }

    public var asArray: [JSONValue]? {
        if case .array(let a) = self { return a }
        return nil
    }

    public var asString: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    public var asInt: Int? {
        if case .int(let i) = self { return i }
        return nil
    }

    public var asDouble: Double? {
        if case .double(let d) = self { return d }
        if case .int(let i)    = self { return Double(i) }
        return nil
    }

    public var asBool: Bool? {
        if case .bool(let b) = self { return b }
        return nil
    }
}
