import Foundation

/// Tipizzazione strict-Sendable di un grafo JSON. Sostituisce
/// `[String: Any]` ovunque il dato debba attraversare un confine
/// `Sendable` (struct conformi a Sendable, actor isolation,
/// Swift 6 strict concurrency).
///
/// Mantiene la stessa flessibilità di `[String: Any]`: i provider
/// possono inserire extension (`enum`, `format`, `examples`,
/// `x-*` field) come nested `.object(...)` / `.array(...)` /
/// scalari senza dover allargare l'enum.
///
/// **NOTA layering**: questo tipo è duplicato in
/// `Sources/DeepSeekKit/Abstractions/JSONValue.swift` con lo
/// stesso identico body. La duplicazione serve a evitare di
/// imporre `DeepSeekKit -> DeepSeekTools` o
/// `DeepSeekTools -> DeepSeekKit` come dipendenza (vedi
/// `Package.swift:54-61`). DeepSeekUI (che importa entrambi i
/// target) può convertire fra le due forme via
/// `foundationValue` / `init(any:)`.
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
        // Bool prima di Int perché in Swift JSONDecoder un Bool
        // non si converte a Int e viceversa — l'ordine fa solo
        // da fast-path.
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

    /// Grafo `Any` compatibile con `JSONSerialization.data(
    /// withJSONObject:)`. Usato ai confini di chi accetta ancora
    /// `[String: Any]` (es. URLSession HTTP bodies, parser
    /// legacy). Localmente non c'è problema di Sendable: il
    /// valore vive dentro una singola funzione, non attraversa
    /// actor.
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

    /// Costruzione da un grafo `Any` arbitrario (es. risultato di
    /// `JSONSerialization.jsonObject(with:)`). Tipi non
    /// riconosciuti diventano `.null` — preferibile a un crash.
    public init(any: Any) {
        if any is NSNull {
            self = .null
            return
        }
        // L'ordine conta: NSNumber può rispondere come Bool E
        // come Int. Controllo Bool tramite il flag `objCType`.
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
