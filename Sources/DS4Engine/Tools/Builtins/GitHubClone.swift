import Foundation
import DS4Core

extension ToolRegistry {
    /// Download a public GitHub repository and import it as the active project
    /// (host pinned to codeload.github.com, arguments validated, size-capped —
    /// see GitHubTool). Returns a compact orientation summary so the model can
    /// explore with the project_* tools instead of reading the repo wholesale.
    static let githubClone = BuiltinTool(
        spec: ToolSpec(name: "github_clone",
                       description: "Download a PUBLIC GitHub repository (HTTPS tarball, no git needed) and import it as the ACTIVE project, replacing the current one. Returns a compact orientation summary: the file tree plus the documentation files to read first. Then explore the structure with project_tree/project_list, locate code with project_find/project_search, and read only the relevant ranges with project_read — never read the whole repository.",
                       parametersJSON: #"{"type":"object","properties":{"repo":{"type":"string","description":"'owner/name' (e.g. \"apple/swift\") or a github.com URL"},"ref":{"type":"string","description":"branch, tag, or commit SHA (default: the repository's default branch)"}},"required":["repo"]}"#),
        run: { argsJSON in
            guard let repo = stringArg(argsJSON, "repo"), !repo.isEmpty else {
                return "Missing 'repo' argument. Example: {\"repo\":\"apple/swift-argument-parser\"}."
            }
            return GitHubTool.run(repoArg: repo, ref: stringArg(argsJSON, "ref"))
        })
}
