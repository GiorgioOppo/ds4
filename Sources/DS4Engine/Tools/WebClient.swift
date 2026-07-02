import Foundation

// Shared HTTP client for the web_search / web_fetch tools. These tools are
// driven by MODEL OUTPUT (attacker-influençable via prompt injection from an
// imported project or a fetched page), so every request goes through an
// SSRF guard: only http/https, and the resolved host must be a PUBLIC address
// — never loopback / private / link-local / ULA. Responses are size- and
// time-capped. Synchronous by design: tools run off the main thread inside the
// InferenceService actor, and BuiltinTool.run is a sync closure.
enum WebClient {
    struct Response { let status: Int; let body: Data; let finalURL: URL; let mime: String }
    enum WebError: Error, CustomStringConvertible {
        case badURL, blockedHost(String), tooLarge, network(String), http(Int)
        var description: String {
            switch self {
            case .badURL: return "URL non valido (usa http:// o https://)"
            case .blockedHost(let h): return "host non consentito: \(h) (loopback/rete privata bloccati)"
            case .tooLarge: return "risposta troppo grande"
            case .network(let m): return "errore di rete: \(m)"
            case .http(let c): return "HTTP \(c)"
            }
        }
    }

    static let maxBytes = 3 * 1024 * 1024        // 3 MB hard cap on any response
    static let timeout: TimeInterval = 20

    /// Fetch a URL after validating scheme + resolving the host to public IPs.
    /// `accept` sets the Accept header; redirects are followed but EACH hop is
    /// re-validated by the delegate (a public URL can 302 to localhost).
    static func get(_ urlString: String, accept: String) throws -> Response {
        guard let url = URL(string: urlString.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
              let host = url.host, !host.isEmpty else {
            throw WebError.badURL
        }
        try assertPublicHost(host)

        var req = URLRequest(url: url, timeoutInterval: timeout)
        req.setValue(accept, forHTTPHeaderField: "Accept")
        req.setValue("DwarfStar/1.0 (+local DeepSeek-V4 agent)", forHTTPHeaderField: "User-Agent")

        let delegate = RedirectGuard()
        let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        let sem = DispatchSemaphore(value: 0)
        var result: Result<Response, Error> = .failure(WebError.network("nessuna risposta"))
        let task = session.dataTask(with: req) { data, resp, err in
            defer { sem.signal() }
            if let err = err {
                // A redirect blocked by the delegate surfaces as a cancel.
                if let ssrf = delegate.blocked { result = .failure(WebError.blockedHost(ssrf)); return }
                result = .failure(WebError.network(err.localizedDescription)); return
            }
            guard let http = resp as? HTTPURLResponse, let data = data else {
                result = .failure(WebError.network("risposta vuota")); return
            }
            if data.count > maxBytes { result = .failure(WebError.tooLarge); return }
            let mime = (http.mimeType ?? "").lowercased()
            result = .success(Response(status: http.statusCode, body: data,
                                       finalURL: http.url ?? url, mime: mime))
        }
        task.resume()
        if sem.wait(timeout: .now() + timeout + 5) == .timedOut {
            task.cancel()
            throw WebError.network("timeout")
        }
        return try result.get()
    }

    /// Throw unless EVERY address `host` resolves to is a public unicast IP.
    static func assertPublicHost(_ host: String) throws {
        let h = host.lowercased()
        if h == "localhost" || h.hasSuffix(".localhost") || h.hasSuffix(".local") ||
           h == "metadata.google.internal" {
            throw WebError.blockedHost(host)
        }
        var hints = addrinfo(ai_flags: 0, ai_family: AF_UNSPEC, ai_socktype: SOCK_STREAM,
                             ai_protocol: 0, ai_addrlen: 0, ai_canonname: nil, ai_addr: nil, ai_next: nil)
        var res: UnsafeMutablePointer<addrinfo>?
        guard getaddrinfo(host, nil, &hints, &res) == 0, let first = res else {
            throw WebError.blockedHost(host)   // unresolvable → refuse
        }
        defer { freeaddrinfo(res) }
        var it: UnsafeMutablePointer<addrinfo>? = first
        var sawAny = false
        while let cur = it {
            sawAny = true
            if let sa = cur.pointee.ai_addr, !isPublic(sa, family: cur.pointee.ai_family) {
                throw WebError.blockedHost(host)
            }
            it = cur.pointee.ai_next
        }
        if !sawAny { throw WebError.blockedHost(host) }
    }

    /// Public unicast? Rejects loopback, private (10/8, 172.16/12, 192.168/16,
    /// 100.64/10 CGNAT), link-local (169.254/16, fe80::/10), ULA (fc00::/7),
    /// unspecified, and IPv6 loopback / v4-mapped private.
    private static func isPublic(_ sa: UnsafePointer<sockaddr>, family: Int32) -> Bool {
        if family == AF_INET {
            let b = withUnsafePointer(to: sa.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { $0.pointee.sin_addr }) {
                $0.withMemoryRebound(to: UInt8.self, capacity: 4) { [$0[0], $0[1], $0[2], $0[3]] }
            }
            switch b[0] {
            case 0, 10, 127: return false                                   // unspecified, private, loopback
            case 100: return !(b[1] >= 64 && b[1] <= 127)                   // 100.64/10 CGNAT
            case 169: return b[1] != 254                                    // 169.254/16 link-local
            case 172: return !(b[1] >= 16 && b[1] <= 31)                    // 172.16/12
            case 192: return b[1] != 168                                    // 192.168/16
            default: return b[0] < 224                                      // reject multicast/reserved ≥224
            }
        }
        if family == AF_INET6 {
            let a = withUnsafePointer(to: sa.withMemoryRebound(to: sockaddr_in6.self, capacity: 1) { $0.pointee.sin6_addr }) {
                $0.withMemoryRebound(to: UInt8.self, capacity: 16) { p in (0..<16).map { p[$0] } }
            }
            if a.allSatisfy({ $0 == 0 }) { return false }                   // ::
            if a[0..<15].allSatisfy({ $0 == 0 }) && a[15] == 1 { return false } // ::1 loopback
            if a[0] == 0xfe && (a[1] & 0xc0) == 0x80 { return false }       // fe80::/10 link-local
            if (a[0] & 0xfe) == 0xfc { return false }                       // fc00::/7 ULA
            // v4-mapped ::ffff:a.b.c.d — re-check the embedded v4.
            if a[0..<10].allSatisfy({ $0 == 0 }) && a[10] == 0xff && a[11] == 0xff {
                let v4 = [a[12], a[13], a[14], a[15]]
                switch v4[0] {
                case 0, 10, 127: return false
                case 169: return v4[1] != 254
                case 172: return !(v4[1] >= 16 && v4[1] <= 31)
                case 192: return v4[1] != 168
                default: return v4[0] < 224
                }
            }
            return true
        }
        return false
    }

    /// URLSession delegate that re-validates the target of every redirect: a
    /// public URL can 302 to http://localhost, so the SSRF guard must run per hop.
    final class RedirectGuard: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
        var blocked: String?
        func urlSession(_ session: URLSession, task: URLSessionTask,
                        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                        completionHandler: @escaping (URLRequest?) -> Void) {
            guard let host = request.url?.host,
                  let scheme = request.url?.scheme?.lowercased(),
                  scheme == "http" || scheme == "https" else {
                blocked = request.url?.host ?? "redirect"; completionHandler(nil); return
            }
            do { try WebClient.assertPublicHost(host); completionHandler(request) }
            catch { blocked = host; completionHandler(nil) }
        }
    }

    /// Strip HTML to readable-ish plain text: drop script/style, tags → spaces,
    /// decode the common entities, collapse whitespace. Not a full parser — good
    /// enough for the model to read a page. `limit` caps the returned length.
    static func htmlToText(_ html: String, limit: Int) -> String {
        var s = html
        for tag in ["script", "style", "noscript", "svg", "head"] {
            s = s.replacingOccurrences(of: "(?s)<\(tag)[^>]*>.*?</\(tag)>", with: " ",
                                       options: .regularExpression)
        }
        s = s.replacingOccurrences(of: "(?s)<!--.*?-->", with: " ", options: .regularExpression)
        s = s.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
        let entities = ["&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
                        "&quot;": "\"", "&#39;": "'", "&apos;": "'", "&mdash;": "—", "&ndash;": "–"]
        for (k, v) in entities { s = s.replacingOccurrences(of: k, with: v) }
        s = s.replacingOccurrences(of: "[ \\t\\x0B\\f\\r]+", with: " ", options: .regularExpression)
        s = s.replacingOccurrences(of: " *\\n *(\\n *)+", with: "\n\n", options: .regularExpression)
        s = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.count > limit { s = String(s.prefix(limit)) + "\n…[troncato]" }
        return s
    }

    /// JSON-escape a string for embedding in a hand-built result object.
    static func jsonEscape(_ s: String) -> String {
        guard let d = try? JSONSerialization.data(withJSONObject: [s]),
              let arr = String(data: d, encoding: .utf8) else { return "\"\"" }
        return String(arr.dropFirst().dropLast())   // strip the [ ]
    }
}
