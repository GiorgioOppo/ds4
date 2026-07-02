import Foundation
import DS4Core

extension ToolRegistry {
    /// Fetch a web page (or JSON/text resource) and return its readable content.
    /// SSRF-guarded (see WebClient): http/https only, public hosts only, size/
    /// time capped. HTML is stripped to plain text; JSON/text is returned as-is.
    static let webFetch = BuiltinTool(
        spec: ToolSpec(name: "web_fetch",
                       description: "Fetch a URL (http/https) and return its readable text content. Use it to read a page found via web_search. Blocks local/private addresses.",
                       parametersJSON: #"{"type":"object","properties":{"url":{"type":"string","description":"absolute http(s) URL"}},"required":["url"]}"#),
        run: { argsJSON in
            guard let url = stringArg(argsJSON, "url"), !url.isEmpty else {
                return #"{"error":"missing 'url' argument"}"#
            }
            do {
                let r = try WebClient.get(url, accept: "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.5")
                guard (200..<300).contains(r.status) else {
                    return #"{"error":"HTTP \#(r.status)","url":"\#(WebClient.jsonEscape(r.finalURL.absoluteString))"}"#
                }
                let raw = String(data: r.body, encoding: .utf8)
                    ?? String(data: r.body, encoding: .isoLatin1) ?? ""
                let isHTML = r.mime.contains("html") || raw.range(of: "(?i)<html|<!doctype html", options: .regularExpression) != nil
                let text = isHTML ? WebClient.htmlToText(raw, limit: 12000)
                                  : (raw.count > 12000 ? String(raw.prefix(12000)) + "\n…[troncato]" : raw)
                return #"{"url":"\#(WebClient.jsonEscape(r.finalURL.absoluteString))","content":"\#(WebClient.jsonEscape(text))"}"#
            } catch {
                return #"{"error":"\#(WebClient.jsonEscape("\(error)"))"}"#
            }
        })
}
