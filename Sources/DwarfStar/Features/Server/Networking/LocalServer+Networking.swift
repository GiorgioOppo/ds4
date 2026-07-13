import Foundation
@preconcurrency import Network
import DS4Core
import DS4Engine

extension LocalServer {
// MARK: Low-level HTTP (async wrappers over NWConnection)

    func send(_ conn: NWConnection, _ data: Data) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            conn.send(content: data, completion: .contentProcessed { error in
                if let error { cont.resume(throwing: error) } else { cont.resume() }
            })
        }
    }

    func receive(_ conn: NWConnection) async throws -> Data? {
        // Cancellation-aware: cancelling the surrounding task (read timeout)
        // cancels the connection, which fails the pending receive and lets the
        // continuation resume — otherwise the read would hang until the peer
        // closes the socket.
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Data?, Error>) in
                conn.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { data, _, isComplete, error in
                    if let error { cont.resume(throwing: error); return }
                    if let data, !data.isEmpty { cont.resume(returning: data) }
                    else { cont.resume(returning: isComplete ? nil : Data()) }
                }
            }
        } onCancel: {
            conn.cancel()
        }
    }

    /// Read a full request within `readTimeout` — a stalled client is dropped
    /// (its socket cancelled) instead of holding the connection open forever.
    func readRequest(_ conn: NWConnection) async throws -> HTTPRequest? {
        try await withThrowingTaskGroup(of: HTTPRequest?.self) { group in
            group.addTask { try await self.readRequestNow(conn) }
            group.addTask {
                try await Task.sleep(nanoseconds: Self.readTimeout)
                throw ServerError.timeout
            }
            defer { group.cancelAll() }
            return try await group.next() ?? nil
        }
    }

    /// Read a full HTTP/1.1 request: headers up to CRLFCRLF, then `Content-Length` body bytes.
    func readRequestNow(_ conn: NWConnection) async throws -> HTTPRequest? {
        var buf = Data()
        let sep = Data("\r\n\r\n".utf8)
        // Read until headers are complete.
        while buf.range(of: sep) == nil {
            guard let chunk = try await receive(conn) else { return nil }
            if chunk.isEmpty { continue }
            buf.append(chunk)
            if buf.count > 8 * 1024 * 1024 { return nil }   // guard runaway headers
        }
        guard let headerEnd = buf.range(of: sep) else { return nil }
        let headerData = buf.subdata(in: buf.startIndex..<headerEnd.lowerBound)
        guard let headerText = String(data: headerData, encoding: .utf8) else { return nil }

        var lines = headerText.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return nil }
        let comps = requestLine.split(separator: " ")
        guard comps.count >= 2 else { return nil }
        let method = String(comps[0])
        let path = String(comps[1].split(separator: "?").first ?? comps[1])

        lines.removeFirst()
        var headers: [String: String] = [:]
        for line in lines {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }
        let contentLength = Int(headers["content-length"] ?? "") ?? 0
        guard contentLength <= Self.maxBodyBytes else { throw ServerError.bodyTooLarge }

        var body = buf.subdata(in: headerEnd.upperBound..<buf.endIndex)
        while body.count < contentLength {
            guard let chunk = try await receive(conn) else { break }
            if chunk.isEmpty { continue }
            body.append(chunk)
            if body.count > Self.maxBodyBytes { throw ServerError.bodyTooLarge }
        }
        return HTTPRequest(method: method, path: path, headers: headers, body: body)
    }
}

