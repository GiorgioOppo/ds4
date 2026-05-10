import Foundation
import PackagePlugin

// Compiles DeepSeekKit/Kernels/*.metal into a default.metallib resource.
//
// Plain `swift build` does not run Apple's Metal toolchain over .metal files
// listed via `.process(...)`; the source files end up copied verbatim into
// the resource bundle and Bundle.module.makeDefaultLibrary fails at runtime
// with MTLLibraryErrorDomain code 6.
//
// This plugin closes that gap: a prebuildCommand runs build_metallib.sh,
// which calls `xcrun metal` + `xcrun metallib` and writes default.metallib
// into the prebuild output directory. SwiftPM then bundles every file in
// that directory as a resource of DeepSeekKit, so Bundle.module finds the
// metallib at runtime.
@main
struct MetalLibPlugin: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) throws -> [Command] {
        guard let target = target as? SourceModuleTarget else { return [] }

        let kernelsDir = target.directory.appending("Kernels")
        let outputDir = context.pluginWorkDirectory.appending("metallib")

        let script = context.package.directory
            .appending("Plugins")
            .appending("MetalLibPlugin")
            .appending("build_metallib.sh")

        return [
            .prebuildCommand(
                displayName: "Compile DeepSeekKit/Kernels into default.metallib",
                executable: Path("/bin/bash"),
                arguments: [
                    script.string,
                    kernelsDir.string,
                    outputDir.string,
                ],
                outputFilesDirectory: outputDir
            )
        ]
    }
}
