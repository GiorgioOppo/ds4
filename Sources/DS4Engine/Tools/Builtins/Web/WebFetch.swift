import Foundation
import DS4Core

extension ToolRegistry {
    /// Fetch a web page (or JSON/text resource) and return its readable content.
    /// SSRF-guarded (see WebClient): http/https only, public hosts only, size/
    /// time capped. HTML is stripped to plain text; JSON/text is returned as-is.
    static let webFetch = BuiltinTool(
        spec: ToolSpec(name: "web_fetch",
                       description: "Fetch a URL (http/https) and return its readable text content. Use it to read a page found via web_search. Long pages are windowed: pass 'offset' (from the previous result's continuation hint) to read the next chunk. Blocks local/private addresses.",
                       parametersJSON: #"{"type":"object","properties":{"url":{"type":"string","description":"absolute http(s) URL"},"offset":{"type":"number","description":"skip this many characters of the extracted text (default 0) — to read a long page in chunks"}},"required":["url"]}"#),
        run: { argsJSON in
            guard let url = stringArg(argsJSON, "url"), !url.isEmpty else {
                return #"{"error":"missing 'url' argument"}"#
            }
            let offset = max(0, intArg(argsJSON, "offset") ?? 0)
            let window = 12000
            do {
                let r = try WebClient.get(url, accept: "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.5")
                guard (200..<300).contains(r.status) else {
                    return #"{"error":"HTTP \#(r.status)","url":"\#(WebClient.jsonEscape(r.finalURL.absoluteString))"}"#
                }
                let raw = String(data: r.body, encoding: .utf8)
                    ?? String(data: r.body, encoding: .isoLatin1) ?? ""
                let isHTML = r.mime.contains("html") || raw.range(of: "(?i)<html|<!doctype html", options: .regularExpression) != nil
                // Body is already capped at WebClient.maxBytes, so extracting the
                // full text (no limit) stays bounded; the window is applied below.
                let full = isHTML ? WebClient.htmlToText(raw, limit: Int.max) : raw
                guard offset < full.count || full.isEmpty else {
                    return #"{"error":"offset \#(offset) is beyond the end (text is \#(full.count) characters)"}"#
                }
                var text = String(full.dropFirst(offset).prefix(window))
                if offset + text.count < full.count {
                    text += "\n…[truncated: call web_fetch again with offset=\(offset + text.count) (text is \(full.count) characters)]"
                }
                return #"{"url":"\#(WebClient.jsonEscape(r.finalURL.absoluteString))","content":"\#(WebClient.jsonEscape(text))"}"#
            } catch {
                return #"{"error":"\#(WebClient.jsonEscape("\(error)"))"}"#
            }
        })
}
