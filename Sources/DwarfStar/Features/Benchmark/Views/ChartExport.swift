import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Saves a benchmark chart as a self-describing PNG (title + context line +
/// chart at a fixed export size) or its underlying data as CSV, via the
/// sandbox-safe save panel. Rendering is forced to light appearance so the
/// exported image is deterministic regardless of the app theme.
@MainActor
enum ChartExport {
    /// Returns nil on success or a message for the benchmark log.
    static func savePNG(title: String, subtitle: String,
                        suggestedFileName: String,
                        chart: some View) -> String? {
        let content = VStack(alignment: .leading, spacing: 10) {
            Text(title).font(.title3.weight(.semibold))
            Text(subtitle).font(.caption).foregroundStyle(.secondary)
            chart.frame(width: 1100, height: 600)
        }
        .padding(24)
        .background(Color.white)
        .environment(\.colorScheme, .light)

        let renderer = ImageRenderer(content: content)
        renderer.scale = 2
        guard let cgImage = renderer.cgImage,
              let data = NSBitmapImageRep(cgImage: cgImage)
                  .representation(using: .png, properties: [:]) else {
            return "PNG export failed: the chart could not be rendered.\n"
        }
        return save(data: data, type: .png, fileName: suggestedFileName + ".png")
    }

    /// Returns nil on success or a message for the benchmark log.
    static func saveCSV(_ csv: String, suggestedFileName: String) -> String? {
        save(data: Data(csv.utf8), type: .commaSeparatedText,
             fileName: suggestedFileName + ".csv")
    }

    /// Timestamped default filename so repeated exports never collide.
    static func stampedName(_ base: String) -> String {
        let format = DateFormatter()
        format.dateFormat = "yyyyMMdd-HHmmss"
        return base + "-" + format.string(from: Date())
    }

    private static func save(data: Data, type: UTType, fileName: String) -> String? {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [type]
        panel.nameFieldStringValue = fileName
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        do {
            try data.write(to: url)
            return nil
        } catch {
            return "Export failed: \(error.localizedDescription)\n"
        }
    }
}
