import Foundation

/// One-call helper to populate a `ToolRegistry` with the native tools
/// shipped by this package. Used by the GUI / CLI to wire a sensible
/// default; callers can still register additional tools (e.g.
/// MCP-backed ones) on the same registry afterwards.
///
/// `includeShell` / `includeNetwork` / `includeRepoClone` let the
/// caller opt out of the most invasive tools — for example, a
/// `.plan`-only agent often wants the network off too. The `lsp`
/// tool is *not* registered by default because its implementation
/// is a stub (`ToolError.notImplemented`); pass `includeStubs: true`
/// to surface it anyway.
public enum DefaultTools {
    public static func standard(planStore: PlanStore,
                                includeShell: Bool = true,
                                includeNetwork: Bool = true,
                                includeRepoClone: Bool = true,
                                includeStubs: Bool = false,
                                includeUnixTools: Bool = false,
                                includeXcodeTools: Bool = false,
                                shellUsesSandbox: Bool = false,
                                shellSandboxProfileBuilder: (@Sendable ([URL]) -> String)? = nil,
                                webSearchProvider: WebSearchProvider? = nil) -> [Tool] {
        var tools: [Tool] = [
            ReadTool(),
            WriteTool(),
            EditTool(),
            GlobTool(),
            GrepTool(),
            ApplyPatchTool(),
            RepoOverviewTool(),
            PlanTool(store: planStore),
            TaskTool(store: planStore),
            TodoTool(store: planStore),
        ]
        if includeShell {
            tools.append(ShellTool(
                useSandbox: shellUsesSandbox,
                profileBuilder: shellSandboxProfileBuilder))
        }
        if includeNetwork {
            tools.append(WebFetchTool())
            // The caller can swap the search backend in via
            // `webSearchProvider`; nil keeps the DuckDuckGo lite
            // scraper default. See TavilyProvider / BraveProvider /
            // SerperProvider in `WebSearchTool.swift`.
            if let provider = webSearchProvider {
                tools.append(WebSearchTool(provider: provider))
            } else {
                tools.append(WebSearchTool())
            }
        }
        if includeRepoClone {
            tools.append(RepoCloneTool())
        }
        if includeStubs {
            tools.append(LSPTool())
        }
        if includeUnixTools {
            tools.append(contentsOf: unixTools())
        }
        if includeXcodeTools {
            tools.append(contentsOf: xcodeTools())
        }
        return tools
    }

    /// The 50-tool Unix toolbox under `Sources/DeepSeekTools/Tools/Unix/`.
    /// Opt-in via `includeUnixTools: true` on `standard(...)` — default
    /// off so the existing 16-tool surface keeps the same defaults for
    /// agents that don't need the extra primitives.
    ///
    /// Surface, by family:
    ///   - Files (10): ls, head, tail, wc, stat, du, basename, dirname,
    ///                 find, which
    ///   - Text (10):  sort, uniq, cut, tr, paste, comm, xxd, md5,
    ///                 sha1, sha256
    ///   - Hash (1):   base64
    ///   - TextBin (3): sed, awk, file
    ///   - Mutate (7): touch, mkdir, cp, mv, rm, ln, chmod
    ///   - Archive (5): tar, gzip, gunzip, zip, unzip
    ///   - System (6): uname, date, env, hostname, whoami, id
    ///   - Process (3): ps, lsof, kill
    ///   - JSON (1):   jq
    ///   - Git (4):    git_status, git_log, git_diff, git_blame
    public static func unixTools() -> [Tool] {
        return [
            // Files
            LsTool(), HeadTool(), TailTool(), WcTool(), StatTool(),
            DuTool(), BasenameTool(), DirnameTool(), FindTool(), WhichTool(),
            // Text
            SortTool(), UniqTool(), CutTool(), TrTool(), PasteTool(),
            CommTool(), XxdTool(), Md5Tool(), Sha1Tool(), Sha256Tool(),
            // Hash / Text via binary
            Base64Tool(), SedTool(), AwkTool(), FileTool(),
            // Mutating
            TouchTool(), MkdirTool(), CpTool(), MvTool(),
            RmTool(), LnTool(), ChmodTool(),
            // Archive
            TarTool(), GzipTool(), GunzipTool(), ZipTool(), UnzipTool(),
            // System
            UnameTool(), DateTool(), EnvTool(),
            HostnameTool(), WhoamiTool(), IdTool(),
            // Process
            PsTool(), LsofTool(), KillTool(),
            // JSON
            JqTool(),
            // Git
            GitStatusTool(), GitLogTool(), GitDiffTool(), GitBlameTool(),
        ]
    }

    /// The 30-tool Xcode / Apple-platform development toolbox under
    /// `Sources/DeepSeekTools/Tools/Xcode/`. Opt-in via
    /// `includeXcodeTools: true` on `standard(...)` — default off so
    /// non-Apple-platform agents and CI without Xcode CLT don't pull
    /// in tools that would fail-by-config.
    ///
    /// Surface, by family:
    ///   - Build (8):     xcodebuild_list / build / test / clean /
    ///                    archive / showsdks / showdestinations /
    ///                    exportarchive
    ///   - SPM (3):       swift_build / test / package
    ///   - Simulator (8): simctl_list / boot / shutdown / install /
    ///                    launch / uninstall / screenshot / erase
    ///   - Device (2):    devicectl_list / install (dangerous)
    ///   - Signing (3):   codesign_verify / display,
    ///                    security_find_identity
    ///   - Mach-O (2):    otool_info, lipo_info
    ///   - Plist (4):     plutil_print / lint, agvtool_version,
    ///                    xcresulttool_get
    ///
    /// Every Xcode-toolchain command goes through `/usr/bin/xcrun`
    /// (see `_Xcrun.swift`) so the active Xcode picked by
    /// `xcode-select -p` is the one that runs. Stable `/usr/bin/X`
    /// binaries (codesign, security, otool, lipo, plutil) call
    /// `UnixBinary.runBinary` directly.
    public static func xcodeTools() -> [Tool] {
        return [
            // Build (8)
            XcodebuildListTool(),
            XcodebuildBuildTool(),
            XcodebuildTestTool(),
            XcodebuildCleanTool(),
            XcodebuildArchiveTool(),
            XcodebuildShowSdksTool(),
            XcodebuildShowDestinationsTool(),
            XcodebuildExportArchiveTool(),
            // SPM (3)
            SwiftBuildTool(),
            SwiftTestTool(),
            SwiftPackageTool(),
            // Simulator (8)
            SimctlListTool(),
            SimctlBootTool(),
            SimctlShutdownTool(),
            SimctlInstallTool(),
            SimctlLaunchTool(),
            SimctlUninstallTool(),
            SimctlScreenshotTool(),
            SimctlEraseTool(),
            // Device (2)
            DevicectlListTool(),
            DevicectlInstallTool(),
            // Signing (3)
            CodesignVerifyTool(),
            CodesignDisplayTool(),
            SecurityFindIdentityTool(),
            // Mach-O inspect (2)
            OtoolInfoTool(),
            LipoInfoTool(),
            // Plist / version / results (4)
            PlutilPrintTool(),
            PlutilLintTool(),
            AgvtoolVersionTool(),
            XcresulttoolGetTool(),
        ]
    }
}
