import Foundation
import DS4Core

extension ToolRegistry {
    static let git = BuiltinTool(
        spec: ToolSpec(name: "git",
                       description: "Run a LOCAL git subcommand in the imported project. Allowed: status, diff, log, show, branch, blame, grep, add, commit, stash, tag, rev-parse, ls-files. No push/pull/network. Example: {\"args\":\"diff --stat\"} or {\"args\":\"commit -am \\\"fix: ...\\\"\"}.",
                       parametersJSON: #"{"type":"object","properties":{"args":{"type":"string","description":"git subcommand and arguments"}},"required":["args"]}"#),
        run: { argsJSON in
            guard let a = stringArg(argsJSON, "args") else { return "Argomento 'args' mancante." }
            return GitTool.run(argsLine: a)
        })
}
