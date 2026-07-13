import Foundation
@preconcurrency import Network
import DS4Core
import DS4Engine

extension LocalServer {
// MARK: Response builders (faithful to ds4_server.c)

    static func corsHeaders(_ cors: Bool) -> String {
        cors ? "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n" : ""
    }

    static func response(_ status: Int, contentType: String?, body: String, cors: Bool) -> Data {
        let reason = statusText(status)
        var head = "HTTP/1.1 \(status) \(reason)\r\n"
        let bodyData = Data(body.utf8)
        if let contentType { head += "Content-Type: \(contentType)\r\n" }
        head += "Content-Length: \(bodyData.count)\r\n"
        head += corsHeaders(cors)
        head += "Connection: close\r\n\r\n"
        var out = Data(head.utf8)
        out.append(bodyData)
        return out
    }

    static func httpError(_ status: Int, _ message: String, cors: Bool) -> Data {
        let payload: [String: Any] = ["error": ["message": message, "type": "invalid_request_error"]]
        let body = (try? JSONSerialization.data(withJSONObject: payload)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        return response(status, contentType: "application/json", body: body, cors: cors)
    }

    static func anthropicError(_ status: Int, _ message: String, cors: Bool) -> Data {
        let payload: [String: Any] = ["type": "error",
                                      "error": ["type": "invalid_request_error", "message": message]]
        let body = (try? JSONSerialization.data(withJSONObject: payload)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        return response(status, contentType: "application/json", body: body, cors: cors)
    }

    static func sseHeader(cors: Bool) -> String {
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n" +
        corsHeaders(cors) + "Connection: close\r\n\r\n"
    }

    static func statusText(_ code: Int) -> String {
        switch code {
        case 200: return "OK"
        case 204: return "No Content"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 404: return "Not Found"
        case 413: return "Payload Too Large"
        case 503: return "Service Unavailable"
        default:  return "OK"
        }
    }

    enum ServerError: Error { case badPort, timeout, bodyTooLarge }
}

