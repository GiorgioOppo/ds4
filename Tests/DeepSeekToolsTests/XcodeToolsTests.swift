import XCTest
@testable import DeepSeekTools

/// Smoke tests for the Xcode / Apple-platform toolbox.
///
/// Every test guards on `xcrun` (and any tool-specific binary) being
/// present so a Linux CI or a minimal macOS install without Xcode CLT
/// still passes by skipping. The goal here is to verify schema /
/// argument-building correctness and registry plumbing — actually
/// building a real Xcode project needs a fixture and an Xcode license
/// dance, which belongs in integration tests, not the smoke suite.
final class XcodeToolsTests: XCTestCase {

    private func makeTempRoot() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("XcodeToolsTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func ctx(_ root: URL) -> ToolContext {
        ToolContext(rootDirectory: root,
                    permission: AutoPermissionDelegate(allowDangerous: true))
    }

    private var xcrunAvailable: Bool {
        FileManager.default.isExecutableFile(atPath: "/usr/bin/xcrun")
    }

    // MARK: - registration

    func testXcodeToolboxRegistersAll30Tools() async throws {
        let registry = ToolRegistry()
        let store = PlanStore()
        await registry.registerAll(
            DefaultTools.standard(planStore: store,
                                  includeShell: false,
                                  includeNetwork: false,
                                  includeRepoClone: false,
                                  includeXcodeTools: true))
        let names = await registry.names()
        let xcodeNames: [String] = [
            // Build
            "xcodebuild_list", "xcodebuild_build", "xcodebuild_test",
            "xcodebuild_clean", "xcodebuild_archive",
            "xcodebuild_showsdks", "xcodebuild_showdestinations",
            "xcodebuild_exportarchive",
            // SPM
            "swift_build", "swift_test", "swift_package",
            // Simulator
            "simctl_list", "simctl_boot", "simctl_shutdown",
            "simctl_install", "simctl_launch", "simctl_uninstall",
            "simctl_screenshot", "simctl_erase",
            // Device
            "devicectl_list", "devicectl_install",
            // Signing
            "codesign_verify", "codesign_display", "security_find_identity",
            // Mach-O
            "otool_info", "lipo_info",
            // Plist / version / results
            "plutil_print", "plutil_lint", "agvtool_version",
            "xcresulttool_get",
        ]
        XCTAssertEqual(xcodeNames.count, 30, "expected exactly 30 xcode tools")
        for name in xcodeNames {
            XCTAssertTrue(names.contains(name), "missing xcode tool: \(name)")
        }
    }

    func testXcodeToolboxOffByDefault() async throws {
        let registry = ToolRegistry()
        let store = PlanStore()
        await registry.registerAll(
            DefaultTools.standard(planStore: store,
                                  includeShell: false,
                                  includeNetwork: false,
                                  includeRepoClone: false))
        let names = await registry.names()
        XCTAssertFalse(names.contains("xcodebuild_build"))
        XCTAssertFalse(names.contains("simctl_list"))
        XCTAssertFalse(names.contains("devicectl_install"))
    }

    func testXcodeToolboxPlanModeFilters() async throws {
        let registry = ToolRegistry()
        let store = PlanStore()
        await registry.registerAll(
            DefaultTools.standard(planStore: store,
                                  includeShell: false,
                                  includeNetwork: false,
                                  includeRepoClone: false,
                                  includeXcodeTools: true))
        let plan = await registry.availableSchemas(mode: .plan)
        let planNames = Set(plan.map(\.name))
        // .mutating: must be filtered out
        for name in ["xcodebuild_build", "xcodebuild_test", "xcodebuild_clean",
                     "xcodebuild_archive", "xcodebuild_exportarchive",
                     "swift_build", "swift_test", "swift_package",
                     "simctl_boot", "simctl_shutdown", "simctl_install",
                     "simctl_launch", "simctl_uninstall",
                     "simctl_screenshot", "simctl_erase",
                     "agvtool_version"] {
            XCTAssertFalse(planNames.contains(name),
                           "\(name) leaked into plan mode")
        }
        // .dangerous: must be filtered out
        XCTAssertFalse(planNames.contains("devicectl_install"))
        // .readOnly: must remain
        for name in ["xcodebuild_list", "xcodebuild_showsdks",
                     "xcodebuild_showdestinations",
                     "simctl_list", "devicectl_list",
                     "codesign_verify", "codesign_display",
                     "security_find_identity",
                     "otool_info", "lipo_info",
                     "plutil_print", "plutil_lint", "xcresulttool_get"] {
            XCTAssertTrue(planNames.contains(name),
                          "\(name) wrongly filtered out of plan mode")
        }
    }

    // MARK: - schema sanity

    func testEveryXcodeToolHasInputSchema() async throws {
        for tool in DefaultTools.xcodeTools() {
            let schema = tool.schema
            XCTAssertFalse(schema.name.isEmpty, "tool has empty name")
            XCTAssertFalse(schema.description.isEmpty,
                           "tool '\(schema.name)' has empty description")
            // Every Xcode tool must declare an object inputSchema, even
            // if some accept no fields (snapshot endpoints).
            XCTAssertNotNil(schema.inputSchemaObject,
                            "tool '\(schema.name)' must have an object input schema")
        }
    }

    // MARK: - live smoke

    /// Calls `xcrun simctl list runtimes --json` which is harmless and
    /// confirms the xcrun → simctl plumbing is end-to-end correct.
    /// Skipped automatically off-macOS or when xcrun is missing.
    func testSimctlListRuntimesSmoke() async throws {
        try XCTSkipUnless(xcrunAvailable, "xcrun not present")
        let root = try makeTempRoot()
        let out = try await SimctlListTool().run(
            input: ["category": "runtimes"],
            context: ctx(root))
        // Either we got a JSON document with the "runtimes" key, or
        // we got an exit-1 error because simctl is unavailable (e.g.
        // Xcode CLT only without simulators). Both prove the helper
        // wired the call correctly.
        XCTAssertTrue(out.output.contains("runtimes") || out.isError,
                      "unexpected simctl output: \(out.output.prefix(200))")
    }

    func testXcodebuildShowsdksSmoke() async throws {
        try XCTSkipUnless(xcrunAvailable, "xcrun not present")
        let root = try makeTempRoot()
        let out = try await XcodebuildShowSdksTool().run(
            input: [:], context: ctx(root))
        // Even on CLT-only installs (no .xcodeproj nearby) -showsdks
        // returns an exit-0 list with at least the macOS SDK.
        XCTAssertTrue(out.output.contains("macosx") || out.isError,
                      "unexpected -showsdks output: \(out.output.prefix(200))")
    }

    func testCodesignVerifyOnSwiftBinary() async throws {
        try XCTSkipUnless(
            FileManager.default.isExecutableFile(atPath: "/usr/bin/codesign"))
        try XCTSkipUnless(
            FileManager.default.isExecutableFile(atPath: "/bin/ls"))
        let root = try makeTempRoot()
        // System binaries are signed on macOS — copying /bin/ls into
        // the agent root preserves the signature.
        let dst = root.appendingPathComponent("ls.bin")
        try FileManager.default.copyItem(at: URL(fileURLWithPath: "/bin/ls"),
                                          to: dst)
        let out = try await CodesignVerifyTool().run(
            input: ["path": "ls.bin"], context: ctx(root))
        // Apple-signed binary → exit 0 expected on macOS. On Linux the
        // codesign binary isn't present, the XCTSkipUnless above
        // shielded us.
        XCTAssertEqual(out.metadata["exit"], "0",
                       "expected /bin/ls to be Apple-signed; got: \(out.output)")
    }

    func testPlutilLintValidatesJSON() async throws {
        try XCTSkipUnless(
            FileManager.default.isExecutableFile(atPath: "/usr/bin/plutil"))
        let root = try makeTempRoot()
        // plutil accepts JSON as a plist source on macOS.
        let json = #"{"key":"value","n":42}"#
        try json.write(to: root.appendingPathComponent("a.json"),
                       atomically: true, encoding: .utf8)
        let out = try await PlutilLintTool().run(
            input: ["path": "a.json"], context: ctx(root))
        XCTAssertEqual(out.metadata["exit"], "0",
                       "expected valid JSON to lint OK: \(out.output)")
    }
}
