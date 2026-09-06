import SwiftUI
import DS4Engine
import DS4Core
import AppKit
import ImageIO

/// A staged text-file attachment shown above the composer, with a remove button.
struct AttachmentChip: View {
    let name: String
    let bytes: Int
    var imageData: Data? = nil
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 4) {
            if let imageData {
                AttachmentThumbnail(data: imageData)
                    .frame(width: 32, height: 32).clipShape(RoundedRectangle(cornerRadius: 4))
            } else {
                Image(systemName: "doc.text").font(.caption2)
            }
            Text(name).font(.caption).lineLimit(1)
            Text(ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file))
                .font(.caption2).foregroundStyle(.secondary)
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help("Rimuovi allegato")
            .accessibilityLabel("Rimuovi \(name)")
        }
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(Color.secondary.opacity(0.12))
        .clipShape(Capsule())
    }
}

/// Decode only a thumbnail for display, keeping full-resolution image memory
/// out of the scrolling transcript.
private func imageThumbnail(_ data: Data) -> NSImage? {
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
          let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 240
          ] as CFDictionary) else { return nil }
    return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
}

private struct AttachmentThumbnail: View {
    let data: Data
    @State private var preview: NSImage?

    var body: some View {
        Group {
            if let preview {
                Image(nsImage: preview).resizable().scaledToFit()
            } else {
                Image(systemName: "photo").foregroundStyle(.secondary)
            }
        }
        .task(id: data) { preview = imageThumbnail(data) }
    }
}

struct ChatImagePreviews: View {
    let images: [ChatImage]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(alignment: .top, spacing: 10) {
                ForEach(Array(images.enumerated()), id: \.offset) { _, image in
                    VStack(alignment: .leading, spacing: 4) {
                        AttachmentThumbnail(data: image.data)
                            .frame(width: 140, height: 110)
                            .background(Color.secondary.opacity(0.08))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .accessibilityLabel("Immagine allegata: \(image.name)")
                        Text(image.name).font(.caption2).lineLimit(1).frame(maxWidth: 140)
                    }
                }
            }
        }
    }
}

/// Filename badges shown under a user message that imported text files.
struct AttachmentBadges: View {
    let names: [String]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(names, id: \.self) { name in
                    Label(name, systemImage: "doc.text")
                        .font(.caption2).lineLimit(1)
                        .padding(.horizontal, 6).padding(.vertical, 3)
                        .background(Color.secondary.opacity(0.12))
                        .clipShape(Capsule())
                }
            }
        }
    }
}
