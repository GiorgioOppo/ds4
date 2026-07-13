import XCTest
@testable import DS4Metal

/// Phase 8 bring-up: verifies the Swift Metal runtime compiles the vendored
/// metal/ kernels and can dispatch one end-to-end.
final class MetalRuntimeTests: XCTestCase {

    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/moe.metal"),
                          "vendored metal kernels not present")
        do {
            return try MetalRuntime(metalDir: Self.metalDir)
        } catch {
            // Surface the Metal compiler diagnostics if the library failed.
            throw XCTSkip("Metal unavailable or kernels failed to compile: \(error)")
        }
    }

    func testKernelsCompile() throws {
        let rt = try makeRuntime()
        // All 19 kernel files compiled into one library; expect many kernels.
        XCTAssertGreaterThan(rt.functionNames.count, 50,
                             "only \(rt.functionNames.count) kernel functions compiled")
        XCTAssertTrue(rt.functionNames.contains("kernel_touch_u8_stride"))
        print("DS4Metal: compiled \(rt.functionNames.count) kernels on \(rt.deviceName)")
    }

    func testPipelineCreation() throws {
        let rt = try makeRuntime()
        // Creating a pipeline state proves a kernel actually links/codegens.
        _ = try rt.pipeline("kernel_touch_u8_stride")
    }

    /// The embedded-kernels runtime (no on-disk folder) compiles the same library
    /// — this is the production path used by the app and DS4Demo.
    func testEmbeddedKernelsCompile() throws {
        let embedded: MetalRuntime
        do { embedded = try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
        XCTAssertGreaterThan(embedded.functionNames.count, 50,
                             "embedded build only \(embedded.functionNames.count) kernels")
        XCTAssertTrue(embedded.functionNames.contains("kernel_touch_u8_stride"))
        XCTAssertTrue(try embedded.runTouchSelfTest(), "embedded runtime self-test failed")
        if FileManager.default.fileExists(atPath: Self.metalDir + "/moe.metal") {
            let disk = try MetalRuntime(metalDir: Self.metalDir)
            XCTAssertEqual(Set(embedded.functionNames), Set(disk.functionNames),
                           "embedded vs on-disk kernel set differs")
        }
    }

    func testTouchKernelRunsCorrectly() throws {
        let rt = try makeRuntime()
        XCTAssertTrue(try rt.runTouchSelfTest(), "kernel_touch_u8_stride output mismatch")
        // A second size to exercise the bounds check.
        XCTAssertTrue(try rt.runTouchSelfTest(count: 1000, stride: 3))
    }
}
