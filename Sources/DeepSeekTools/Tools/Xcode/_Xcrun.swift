import Foundation

/// Thin convenience over `UnixBinary.runBinary` for tools that go
/// through `/usr/bin/xcrun`. xcrun resolves the requested tool from
/// the active Xcode toolchain (`xcode-select -p`), so the model always
/// hits the same SDK its xcodebuild sees — important when multiple
/// Xcodes are installed side-by-side.
///
/// Used by `xcodebuild`, `swift`, `simctl`, `devicectl`, `xcresulttool`,
/// `agvtool`. Tools that live at stable `/usr/bin/<name>` paths
/// (`codesign`, `security`, `otool`, `lipo`, `plutil`) call
/// `UnixBinary.runBinary` directly — no xcrun indirection needed.
public enum Xcrun {
    public static let path = "/usr/bin/xcrun"

    public static func run(tool: String,
                           arguments: [String],
                           context: ToolContext,
                           cwd: URL? = nil,
                           timeout: TimeInterval = 60,
                           outputCap: Int = UnixBinary.defaultOutputCap,
                           separateStreams: Bool = false) async throws
        -> ToolOutput
    {
        return try await UnixBinary.runBinary(
            launchPath: path,
            arguments: [tool] + arguments,
            context: context,
            cwd: cwd,
            timeout: timeout,
            outputCap: outputCap,
            separateStreams: separateStreams)
    }
}
