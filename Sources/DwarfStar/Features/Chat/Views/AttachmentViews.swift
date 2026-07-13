import SwiftUI
import DS4Engine
import DS4Core

/// A staged text-file attachment shown above the composer, with a remove button.
struct AttachmentChip: View {
    let name: String
    let bytes: Int
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "doc.text").font(.caption2)
            Text(name).font(.caption).lineLimit(1)
            Text(ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file))
                .font(.caption2).foregroundStyle(.secondary)
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help("Remove attachment")
        }
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(Color.secondary.opacity(0.12))
        .clipShape(Capsule())
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

