import Foundation

/// Minimal Jinja2-compatible template engine — just enough to render the
/// `chat_template` field that HuggingFace ships in `tokenizer_config.json`
/// for Mistral / Llama / Gemma / Qwen / ChatML and similar models.
///
/// Supported surface:
///   - `{{ var }}` interpolation with chained `.field`/`[index]` access
///   - `{% if cond %}…{% elif cond %}…{% else %}…{% endif %}`
///   - `{% for item in iterable %}…{% endfor %}` with `loop.last`,
///     `loop.first`, `loop.index0` (and `loop.index` = index0 + 1)
///   - `{% set name = expr %}` (simple top-level assignment)
///   - `{{ expr | filter | filter(arg) }}` with `trim`, `lower`,
///     `upper`, `length`, `default(x)`
///   - Operators `==`, `!=`, `<`, `>`, `<=`, `>=`, `and`, `or`, `not`,
///     `in`, `not in`
///   - Negative list indices (`messages[-1]`)
///   - Whitespace trim markers `{%- … -%}` and `{{- … -}}`
///   - `raise_exception("msg")` → throws `ChatTemplateError.templateRaise`
///
/// NOT supported (will throw `ChatTemplateError.unsupportedFeature`):
///   - `{% macro %}`, `{% include %}`, `{% extends %}`, `{% block %}`
///   - User-defined filters or tests
///   - The `~` string-concat operator (use `+` or interpolate instead)
///
/// The engine is single-pass interpret-on-AST. It is fast enough for
/// chat templates (typically 50-200 nodes per call) but not designed
/// for large document rendering.
public enum JinjaValue: CustomStringConvertible, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case list([JinjaValue])
    case dict([String: JinjaValue])

    public var truthy: Bool {
        switch self {
        case .null:                 return false
        case .bool(let b):          return b
        case .int(let i):           return i != 0
        case .double(let d):        return d != 0
        case .string(let s):        return !s.isEmpty
        case .list(let l):          return !l.isEmpty
        case .dict(let d):          return !d.isEmpty
        }
    }

    public var description: String {
        switch self {
        case .null:                 return ""
        case .bool(let b):          return b ? "true" : "false"
        case .int(let i):           return String(i)
        case .double(let d):        return String(d)
        case .string(let s):        return s
        case .list(let l):          return "[" + l.map(\.description).joined(separator: ", ") + "]"
        case .dict(let d):
            let inner = d.map { "\"\($0.key)\": \($0.value.description)" }.joined(separator: ", ")
            return "{" + inner + "}"
        }
    }

    /// Equality used by the `==` / `!=` / `in` operators. Tolerates
    /// int/double cross-comparisons.
    static func == (l: JinjaValue, r: JinjaValue) -> Bool {
        switch (l, r) {
        case (.null, .null):                              return true
        case (.bool(let a), .bool(let b)):                return a == b
        case (.int(let a), .int(let b)):                  return a == b
        case (.double(let a), .double(let b)):            return a == b
        case (.int(let a), .double(let b)):               return Double(a) == b
        case (.double(let a), .int(let b)):               return a == Double(b)
        case (.string(let a), .string(let b)):            return a == b
        default: return false
        }
    }
}

// MARK: - Tokeniser

private enum JToken: Equatable {
    case literal(String)        // raw text between tags
    case expr(String)           // {{ ... }} content (already stripped)
    case stmt(String)           // {% ... %} content (already stripped)
}

private enum JLex {
    /// Lex a template into a flat token list, applying whitespace-trim
    /// markers (`{%-`, `-%}`, `{{-`, `-}}`) so the upstream parser
    /// doesn't have to think about them.
    static func tokenize(_ src: String) throws -> [JToken] {
        var out: [JToken] = []
        var cursor = src.startIndex
        var pendingLiteral = ""

        while cursor < src.endIndex {
            // Find the next tag boundary.
            let openExpr = src.range(of: "{{", range: cursor..<src.endIndex)
            let openStmt = src.range(of: "{%", range: cursor..<src.endIndex)
            // Pick the nearest one.
            var nextOpen: Range<String.Index>? = nil
            var isExpr = false
            if let e = openExpr, let s = openStmt {
                if e.lowerBound < s.lowerBound { nextOpen = e; isExpr = true }
                else { nextOpen = s; isExpr = false }
            } else if let e = openExpr {
                nextOpen = e; isExpr = true
            } else if let s = openStmt {
                nextOpen = s; isExpr = false
            }

            guard let open = nextOpen else {
                pendingLiteral += String(src[cursor..<src.endIndex])
                break
            }

            // Literal between cursor and the tag.
            pendingLiteral += String(src[cursor..<open.lowerBound])

            // Trim leading whitespace from the literal if the tag uses `-`.
            let isTrimLeft: Bool = {
                let after = open.upperBound
                guard after < src.endIndex else { return false }
                return src[after] == "-"
            }()
            if isTrimLeft {
                while let last = pendingLiteral.last, last.isWhitespace {
                    pendingLiteral.removeLast()
                }
            }

            // Find the closing tag.
            let closeMarker = isExpr ? "}}" : "%}"
            guard let close = src.range(of: closeMarker, range: open.upperBound..<src.endIndex) else {
                throw ChatTemplateError.parseFailure("unterminated tag opened with \(isExpr ? "{{" : "{%")")
            }

            // Extract the body, account for `-` trim markers on either side.
            var bodyStart = open.upperBound
            if isTrimLeft { bodyStart = src.index(after: bodyStart) }
            var bodyEnd = close.lowerBound
            let isTrimRight: Bool = {
                let before = src.index(before: bodyEnd)
                return before >= bodyStart && src[before] == "-"
            }()
            if isTrimRight { bodyEnd = src.index(before: bodyEnd) }

            if !pendingLiteral.isEmpty {
                out.append(.literal(pendingLiteral))
                pendingLiteral = ""
            }
            let body = String(src[bodyStart..<bodyEnd]).trimmingCharacters(in: .whitespacesAndNewlines)
            if isExpr {
                out.append(.expr(body))
            } else {
                out.append(.stmt(body))
            }

            cursor = close.upperBound
            // If we trimmed the right side, eat all whitespace after the
            // closing tag too.
            if isTrimRight {
                while cursor < src.endIndex, src[cursor].isWhitespace {
                    cursor = src.index(after: cursor)
                }
            }
        }

        if !pendingLiteral.isEmpty {
            out.append(.literal(pendingLiteral))
        }
        return out
    }
}

// MARK: - AST

private indirect enum JNode {
    case literal(String)
    case expr(JExpr)
    case ifNode([(JExpr, [JNode])], [JNode]?)    // (cond, body)* + optional else body
    case forNode(loopVar: String, iter: JExpr, body: [JNode])
    case setNode(name: String, value: JExpr)
}

private indirect enum JExpr {
    case literal(JinjaValue)
    case variable(String)                 // bare identifier
    case attr(JExpr, String)              // a.b
    case subscriptOp(JExpr, JExpr)        // a[b]
    case call(JExpr, [JExpr])             // f(args)
    case filter(JExpr, String, [JExpr])   // expr | name(args)
    case unary(String, JExpr)             // not x
    case binary(JExpr, String, JExpr)     // a op b
}

private struct JParser {
    let tokens: [JToken]
    var i: Int = 0

    /// Parse the full token stream into a node list. Stops at the end
    /// of the stream or when one of `stops` is seen at the top of the
    /// current statement; in that case the stop token is left
    /// unconsumed for the caller to inspect.
    mutating func parseUntil(_ stops: Set<String>) throws -> [JNode] {
        var nodes: [JNode] = []
        while i < tokens.count {
            switch tokens[i] {
            case .literal(let s):
                nodes.append(.literal(s))
                i += 1
            case .expr(let body):
                let parsed = try JExprParser.parse(body)
                nodes.append(.expr(parsed))
                i += 1
            case .stmt(let body):
                let first = body.split(separator: " ", maxSplits: 1).first.map(String.init) ?? ""
                if stops.contains(first) {
                    return nodes
                }
                let node = try parseStmt(body: body)
                nodes.append(node)
            }
        }
        return nodes
    }

    private mutating func parseStmt(body: String) throws -> JNode {
        let stripped = body.trimmingCharacters(in: .whitespacesAndNewlines)
        let firstWord = stripped.split(separator: " ", maxSplits: 1).first.map(String.init) ?? ""
        switch firstWord {
        case "if":
            return try parseIfBlock(body: stripped)
        case "for":
            return try parseForBlock(body: stripped)
        case "set":
            i += 1
            return try parseSet(body: stripped)
        case "macro", "include", "extends", "block":
            throw ChatTemplateError.unsupportedFeature("statement: \(firstWord)")
        default:
            throw ChatTemplateError.parseFailure("unknown statement: \(stripped)")
        }
    }

    private mutating func parseIfBlock(body: String) throws -> JNode {
        let condStr = String(body.dropFirst("if".count)).trimmingCharacters(in: .whitespaces)
        let firstCond = try JExprParser.parse(condStr)
        i += 1
        var branches: [(JExpr, [JNode])] = []
        var elseBody: [JNode]? = nil
        let firstBody = try parseUntil(["elif", "else", "endif"])
        branches.append((firstCond, firstBody))
        while i < tokens.count {
            guard case .stmt(let stmtBody) = tokens[i] else { break }
            let trimmed = stmtBody.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed == "endif" {
                i += 1
                return .ifNode(branches, elseBody)
            } else if trimmed.hasPrefix("elif") {
                let cs = String(trimmed.dropFirst("elif".count)).trimmingCharacters(in: .whitespaces)
                let c = try JExprParser.parse(cs)
                i += 1
                let b = try parseUntil(["elif", "else", "endif"])
                branches.append((c, b))
            } else if trimmed == "else" {
                i += 1
                elseBody = try parseUntil(["endif"])
            } else {
                break
            }
        }
        throw ChatTemplateError.parseFailure("unterminated {% if %} block")
    }

    private mutating func parseForBlock(body: String) throws -> JNode {
        // body: "for X in EXPR"
        let rest = String(body.dropFirst("for".count)).trimmingCharacters(in: .whitespaces)
        guard let inRange = rest.range(of: " in ") else {
            throw ChatTemplateError.parseFailure("malformed {% for %}: \(body)")
        }
        let loopVar = String(rest[..<inRange.lowerBound]).trimmingCharacters(in: .whitespaces)
        let iterStr = String(rest[inRange.upperBound...]).trimmingCharacters(in: .whitespaces)
        let iter = try JExprParser.parse(iterStr)
        i += 1
        let body = try parseUntil(["endfor"])
        guard i < tokens.count, case .stmt(let endBody) = tokens[i],
              endBody.trimmingCharacters(in: .whitespacesAndNewlines) == "endfor" else {
            throw ChatTemplateError.parseFailure("unterminated {% for %} block")
        }
        i += 1
        return .forNode(loopVar: loopVar, iter: iter, body: body)
    }

    private mutating func parseSet(body: String) throws -> JNode {
        // body: "set NAME = EXPR"
        let rest = String(body.dropFirst("set".count)).trimmingCharacters(in: .whitespaces)
        guard let eq = rest.firstIndex(of: "=") else {
            throw ChatTemplateError.parseFailure("malformed {% set %}: \(body)")
        }
        let name = String(rest[..<eq]).trimmingCharacters(in: .whitespaces)
        let exprStr = String(rest[rest.index(after: eq)...]).trimmingCharacters(in: .whitespaces)
        let expr = try JExprParser.parse(exprStr)
        return .setNode(name: name, value: expr)
    }
}

// MARK: - Expression parser

private struct JExprParser {
    private let src: String
    private var idx: String.Index

    private init(_ s: String) {
        self.src = s
        self.idx = s.startIndex
    }

    static func parse(_ s: String) throws -> JExpr {
        var p = JExprParser(s)
        let expr = try p.parsePipe()
        p.skipWS()
        if p.idx != p.src.endIndex {
            throw ChatTemplateError.parseFailure("trailing input in expression: \(s)")
        }
        return expr
    }

    // pipe := or ( '|' filter )*
    private mutating func parsePipe() throws -> JExpr {
        var lhs = try parseOr()
        while consume("|") {
            skipWS()
            let name = try readIdentifier()
            var args: [JExpr] = []
            skipWS()
            if consume("(") {
                args = try readArgList()
            }
            lhs = .filter(lhs, name, args)
        }
        return lhs
    }

    private mutating func parseOr() throws -> JExpr {
        var lhs = try parseAnd()
        while consumeKeyword("or") {
            let rhs = try parseAnd()
            lhs = .binary(lhs, "or", rhs)
        }
        return lhs
    }

    private mutating func parseAnd() throws -> JExpr {
        var lhs = try parseNot()
        while consumeKeyword("and") {
            let rhs = try parseNot()
            lhs = .binary(lhs, "and", rhs)
        }
        return lhs
    }

    private mutating func parseNot() throws -> JExpr {
        skipWS()
        if consumeKeyword("not") {
            let inner = try parseCompare()
            return .unary("not", inner)
        }
        return try parseCompare()
    }

    private mutating func parseCompare() throws -> JExpr {
        var lhs = try parseAddSub()
        skipWS()
        // Multi-char compare ops first.
        let ops = ["==", "!=", "<=", ">=", "<", ">"]
        for op in ops {
            if consume(op) {
                let rhs = try parseAddSub()
                return .binary(lhs, op, rhs)
            }
        }
        if consumeKeyword("in") {
            let rhs = try parseAddSub()
            return .binary(lhs, "in", rhs)
        }
        if consumeKeywordSequence(["not", "in"]) {
            let rhs = try parseAddSub()
            return .binary(lhs, "not in", rhs)
        }
        return lhs
    }

    private mutating func parseAddSub() throws -> JExpr {
        var lhs = try parseMulDiv()
        while true {
            skipWS()
            if consume("+") {
                let rhs = try parseMulDiv()
                lhs = .binary(lhs, "+", rhs)
            } else if consume("-") {
                let rhs = try parseMulDiv()
                lhs = .binary(lhs, "-", rhs)
            } else {
                break
            }
        }
        return lhs
    }

    private mutating func parseMulDiv() throws -> JExpr {
        var lhs = try parseUnary()
        while true {
            skipWS()
            if consume("*") {
                let rhs = try parseUnary()
                lhs = .binary(lhs, "*", rhs)
            } else if consume("/") {
                let rhs = try parseUnary()
                lhs = .binary(lhs, "/", rhs)
            } else {
                break
            }
        }
        return lhs
    }

    private mutating func parseUnary() throws -> JExpr {
        skipWS()
        if consume("-") {
            let inner = try parsePostfix()
            return .unary("-", inner)
        }
        return try parsePostfix()
    }

    private mutating func parsePostfix() throws -> JExpr {
        var node = try parseAtom()
        while true {
            skipWS()
            if consume(".") {
                skipWS()
                let name = try readIdentifier()
                node = .attr(node, name)
            } else if consume("[") {
                let idxExpr = try parsePipe()
                skipWS()
                guard consume("]") else {
                    throw ChatTemplateError.parseFailure("missing ']' after subscript")
                }
                node = .subscriptOp(node, idxExpr)
            } else if consume("(") {
                let args = try readArgList()
                node = .call(node, args)
            } else {
                break
            }
        }
        return node
    }

    private mutating func parseAtom() throws -> JExpr {
        skipWS()
        guard idx < src.endIndex else {
            throw ChatTemplateError.parseFailure("unexpected end of expression")
        }
        let c = src[idx]
        if c == "'" || c == "\"" {
            return .literal(.string(try readString(quote: c)))
        }
        if c.isNumber {
            return .literal(try readNumber())
        }
        if c == "(" {
            idx = src.index(after: idx)
            let inner = try parsePipe()
            skipWS()
            guard consume(")") else {
                throw ChatTemplateError.parseFailure("missing ')'")
            }
            return inner
        }
        if c == "[" {
            // List literal.
            idx = src.index(after: idx)
            var items: [JExpr] = []
            skipWS()
            if !consume("]") {
                items.append(try parsePipe())
                while true {
                    skipWS()
                    if consume(",") {
                        items.append(try parsePipe())
                    } else if consume("]") {
                        break
                    } else {
                        throw ChatTemplateError.parseFailure("malformed list literal")
                    }
                }
            }
            return .call(.variable("__list__"), items)
        }
        // Identifier (or keyword like true/false/none).
        let name = try readIdentifier()
        switch name {
        case "true":  return .literal(.bool(true))
        case "false": return .literal(.bool(false))
        case "none", "None", "null":  return .literal(.null)
        default:      return .variable(name)
        }
    }

    private mutating func readArgList() throws -> [JExpr] {
        var args: [JExpr] = []
        skipWS()
        if consume(")") { return args }
        args.append(try parsePipe())
        while true {
            skipWS()
            if consume(",") {
                args.append(try parsePipe())
            } else if consume(")") {
                return args
            } else {
                throw ChatTemplateError.parseFailure("malformed argument list")
            }
        }
    }

    // MARK: lexical helpers

    private mutating func skipWS() {
        while idx < src.endIndex, src[idx].isWhitespace {
            idx = src.index(after: idx)
        }
    }

    private mutating func consume(_ s: String) -> Bool {
        let cur = idx
        for ch in s {
            guard cur != src.endIndex, src[idx] == ch else {
                idx = cur
                return false
            }
            idx = src.index(after: idx)
        }
        return true
    }

    private mutating func consumeKeyword(_ kw: String) -> Bool {
        let save = idx
        skipWS()
        var probe = idx
        for ch in kw {
            guard probe != src.endIndex, src[probe] == ch else {
                idx = save
                return false
            }
            probe = src.index(after: probe)
        }
        // Word boundary.
        if probe != src.endIndex, src[probe].isLetter || src[probe].isNumber || src[probe] == "_" {
            idx = save
            return false
        }
        idx = probe
        return true
    }

    private mutating func consumeKeywordSequence(_ words: [String]) -> Bool {
        let save = idx
        for w in words {
            if !consumeKeyword(w) {
                idx = save
                return false
            }
        }
        return true
    }

    private mutating func readIdentifier() throws -> String {
        skipWS()
        guard idx < src.endIndex, src[idx].isLetter || src[idx] == "_" else {
            throw ChatTemplateError.parseFailure("expected identifier")
        }
        var name = ""
        while idx < src.endIndex, src[idx].isLetter || src[idx].isNumber || src[idx] == "_" {
            name.append(src[idx])
            idx = src.index(after: idx)
        }
        return name
    }

    private mutating func readString(quote: Character) throws -> String {
        idx = src.index(after: idx)
        var out = ""
        while idx < src.endIndex {
            let ch = src[idx]
            if ch == "\\", let next = src.index(idx, offsetBy: 1, limitedBy: src.endIndex), next < src.endIndex {
                let esc = src[next]
                switch esc {
                case "n":  out.append("\n")
                case "t":  out.append("\t")
                case "r":  out.append("\r")
                case "\\": out.append("\\")
                case quote: out.append(quote)
                default: out.append(esc)
                }
                idx = src.index(after: next)
                continue
            }
            if ch == quote {
                idx = src.index(after: idx)
                return out
            }
            out.append(ch)
            idx = src.index(after: idx)
        }
        throw ChatTemplateError.parseFailure("unterminated string literal")
    }

    private mutating func readNumber() throws -> JinjaValue {
        var s = ""
        var sawDot = false
        while idx < src.endIndex {
            let c = src[idx]
            if c == "." {
                if sawDot { break }
                sawDot = true
                s.append(c)
                idx = src.index(after: idx)
            } else if c.isNumber {
                s.append(c)
                idx = src.index(after: idx)
            } else { break }
        }
        if sawDot, let d = Double(s) { return .double(d) }
        if let i = Int(s) { return .int(i) }
        throw ChatTemplateError.parseFailure("malformed number: \(s)")
    }
}

// MARK: - Renderer

public struct JinjaTemplate {
    private let nodes: [JNode]

    public init(_ source: String) throws {
        let toks = try JLex.tokenize(source)
        var parser = JParser(tokens: toks)
        self.nodes = try parser.parseUntil([])
    }

    public func render(context: [String: JinjaValue]) throws -> String {
        var scope = context
        return try Self.renderNodes(nodes, scope: &scope)
    }

    private static func renderNodes(_ nodes: [JNode], scope: inout [String: JinjaValue]) throws -> String {
        var out = ""
        for n in nodes {
            switch n {
            case .literal(let s):
                out += s
            case .expr(let e):
                let v = try evalExpr(e, scope: scope)
                out += v.description
            case .ifNode(let branches, let elseBody):
                var rendered = false
                for (cond, body) in branches {
                    let c = try evalExpr(cond, scope: scope)
                    if c.truthy {
                        out += try renderNodes(body, scope: &scope)
                        rendered = true
                        break
                    }
                }
                if !rendered, let elseBody = elseBody {
                    out += try renderNodes(elseBody, scope: &scope)
                }
            case .forNode(let loopVar, let iter, let body):
                let it = try evalExpr(iter, scope: scope)
                guard case .list(let items) = it else {
                    if case .null = it { continue }
                    throw ChatTemplateError.parseFailure("for: iterable is not a list (got \(it))")
                }
                let savedLoop = scope["loop"]
                let savedVar = scope[loopVar]
                for (k, v) in items.enumerated() {
                    scope[loopVar] = v
                    scope["loop"] = .dict([
                        "index0": .int(k),
                        "index":  .int(k + 1),
                        "first":  .bool(k == 0),
                        "last":   .bool(k == items.count - 1),
                        "length": .int(items.count),
                    ])
                    out += try renderNodes(body, scope: &scope)
                }
                scope["loop"] = savedLoop
                scope[loopVar] = savedVar
            case .setNode(let name, let valueExpr):
                scope[name] = try evalExpr(valueExpr, scope: scope)
            }
        }
        return out
    }

    private static func evalExpr(_ e: JExpr, scope: [String: JinjaValue]) throws -> JinjaValue {
        switch e {
        case .literal(let v): return v
        case .variable(let n):
            if n == "__list__" { return .list([]) }  // unused safety
            return scope[n] ?? .null
        case .attr(let obj, let name):
            let o = try evalExpr(obj, scope: scope)
            if case .dict(let d) = o {
                return d[name] ?? .null
            }
            return .null
        case .subscriptOp(let obj, let key):
            let o = try evalExpr(obj, scope: scope)
            let k = try evalExpr(key, scope: scope)
            return subscriptValue(o, k)
        case .call(let f, let args):
            // Only two callables we recognise: __list__ (list literal
            // sugar) and raise_exception(...).
            if case .variable(let name) = f {
                let argVals = try args.map { try evalExpr($0, scope: scope) }
                switch name {
                case "__list__":
                    return .list(argVals)
                case "raise_exception":
                    let msg = argVals.first?.description ?? ""
                    throw ChatTemplateError.templateRaise(msg)
                default:
                    throw ChatTemplateError.unsupportedFeature("call to \(name)")
                }
            }
            throw ChatTemplateError.unsupportedFeature("call on non-identifier")
        case .filter(let inner, let name, let args):
            let v = try evalExpr(inner, scope: scope)
            let argVals = try args.map { try evalExpr($0, scope: scope) }
            return try applyFilter(name: name, value: v, args: argVals)
        case .unary(let op, let inner):
            let v = try evalExpr(inner, scope: scope)
            switch op {
            case "not": return .bool(!v.truthy)
            case "-":
                if case .int(let i) = v { return .int(-i) }
                if case .double(let d) = v { return .double(-d) }
                return .null
            default: return .null
            }
        case .binary(let l, let op, let r):
            let lv = try evalExpr(l, scope: scope)
            let rv = try evalExpr(r, scope: scope)
            return try applyBinary(op: op, l: lv, r: rv)
        }
    }

    private static func subscriptValue(_ obj: JinjaValue, _ key: JinjaValue) -> JinjaValue {
        switch (obj, key) {
        case (.list(let items), .int(let i)):
            let idx = i < 0 ? items.count + i : i
            return (idx >= 0 && idx < items.count) ? items[idx] : .null
        case (.dict(let d), .string(let s)):
            return d[s] ?? .null
        case (.string(let s), .int(let i)):
            let arr = Array(s)
            let idx = i < 0 ? arr.count + i : i
            return (idx >= 0 && idx < arr.count) ? .string(String(arr[idx])) : .null
        default: return .null
        }
    }

    private static func applyFilter(name: String, value: JinjaValue, args: [JinjaValue]) throws -> JinjaValue {
        switch name {
        case "trim":
            if case .string(let s) = value {
                return .string(s.trimmingCharacters(in: .whitespacesAndNewlines))
            }
            return value
        case "lower":
            if case .string(let s) = value { return .string(s.lowercased()) }
            return value
        case "upper":
            if case .string(let s) = value { return .string(s.uppercased()) }
            return value
        case "length":
            if case .string(let s) = value { return .int(s.count) }
            if case .list(let l) = value   { return .int(l.count) }
            if case .dict(let d) = value   { return .int(d.count) }
            return .int(0)
        case "default":
            if case .null = value           { return args.first ?? .null }
            if case .string(let s) = value, s.isEmpty { return args.first ?? .null }
            return value
        case "string":
            return .string(value.description)
        case "tojson":
            // Best-effort JSON serialisation for common types.
            return .string(toJSON(value))
        default:
            throw ChatTemplateError.unsupportedFeature("filter \(name)")
        }
    }

    private static func toJSON(_ v: JinjaValue) -> String {
        switch v {
        case .null: return "null"
        case .bool(let b): return b ? "true" : "false"
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .string(let s):
            return "\"" + s.replacingOccurrences(of: "\\", with: "\\\\")
                          .replacingOccurrences(of: "\"", with: "\\\"")
                          .replacingOccurrences(of: "\n", with: "\\n") + "\""
        case .list(let l):
            return "[" + l.map { toJSON($0) }.joined(separator: ", ") + "]"
        case .dict(let d):
            let entries = d.map { "\"\($0.key)\": \(toJSON($0.value))" }.joined(separator: ", ")
            return "{" + entries + "}"
        }
    }

    private static func applyBinary(op: String, l: JinjaValue, r: JinjaValue) throws -> JinjaValue {
        switch op {
        case "+":
            if case .string(let a) = l, case .string(let b) = r { return .string(a + b) }
            if case .int(let a) = l, case .int(let b) = r { return .int(a + b) }
            if case .double(let a) = l, case .double(let b) = r { return .double(a + b) }
            return .null
        case "-":
            if case .int(let a) = l, case .int(let b) = r { return .int(a - b) }
            if case .double(let a) = l, case .double(let b) = r { return .double(a - b) }
            return .null
        case "*":
            if case .int(let a) = l, case .int(let b) = r { return .int(a * b) }
            if case .double(let a) = l, case .double(let b) = r { return .double(a * b) }
            return .null
        case "/":
            if case .int(let a) = l, case .int(let b) = r, b != 0 { return .int(a / b) }
            if case .double(let a) = l, case .double(let b) = r, b != 0 { return .double(a / b) }
            return .null
        case "==": return .bool(l == r)
        case "!=": return .bool(!(l == r))
        case "<", ">", "<=", ">=":
            return .bool(compareNumbers(l: l, r: r, op: op))
        case "and": return .bool(l.truthy && r.truthy)
        case "or":  return .bool(l.truthy || r.truthy)
        case "in":
            if case .list(let items) = r {
                return .bool(items.contains { $0 == l })
            }
            if case .dict(let d) = r, case .string(let s) = l {
                return .bool(d.keys.contains(s))
            }
            if case .string(let s) = r, case .string(let needle) = l {
                return .bool(s.contains(needle))
            }
            return .bool(false)
        case "not in":
            let inResult = try applyBinary(op: "in", l: l, r: r)
            if case .bool(let b) = inResult { return .bool(!b) }
            return .bool(true)
        default:
            throw ChatTemplateError.unsupportedFeature("operator \(op)")
        }
    }

    private static func compareNumbers(l: JinjaValue, r: JinjaValue, op: String) -> Bool {
        func num(_ v: JinjaValue) -> Double? {
            switch v {
            case .int(let i):    return Double(i)
            case .double(let d): return d
            default:             return nil
            }
        }
        guard let a = num(l), let b = num(r) else { return false }
        switch op {
        case "<":  return a <  b
        case ">":  return a >  b
        case "<=": return a <= b
        case ">=": return a >= b
        default:   return false
        }
    }
}
