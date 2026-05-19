import SwiftUI
import DeepSeekKit
#if canImport(AppKit)
import AppKit
#endif

/// Modal sheet that drives the "vectorize a text document" flow.
///
/// Step 1 covers everything up to and including the tokenisation phase:
/// pick a file → preview metadata → run the BPE tokenizer over the
/// whole text → persist the resulting Int32 token sequence as the
/// library's `<id>.tokens` payload.
///
/// Step 3 will extend the same sheet with the prefill phase: feed
/// those tokens through `Transformer.forward(startPos: 0)` while
/// dumping the per-layer KV cache into `<id>.vec`, so subsequent
/// chats can rehydrate the cache without re-doing the prefill.
struct ImportDocumentSheet: View {
    @ObservedObject var library: DocumentLibrary
    let service: InferenceService

    @Environment(\.dismiss) private var dismiss

    @State private var phase: Phase = .picking
    @State private var pickedURL: URL?
    @State private var displayName: String = ""
    @State private var sourceText: String = ""
    @State private var sourceByteCount: Int = 0
    @State private var tokenCount: Int = 0
    @State private var errorMessage: String?

    enum Phase: Equatable {
        case picking            // no file chosen yet
        case ready              // file loaded, ready to vectorize
        case tokenizing         // BPE encode running
        case saving             // writing tokens to disk
        case done(UUID)         // success; carry the created id for "open in chat" later
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            Divider()
            content
            Spacer()
            Divider()
            footer
        }
        .padding(20)
        .frame(width: 520, height: 380)
    }

    // MARK: - sections

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Import a text document")
                .font(.title3.bold())
            Text("Tokenize the file and add it to the global library. " +
                  "Step 3 will additionally precompute its KV cache so " +
                  "the model can use it as instant context.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var content: some View {
        switch phase {
        case .picking:
            VStack(alignment: .leading, spacing: 12) {
                Button {
                    pickFile()
                } label: {
                    Label("Choose file…", systemImage: "doc.text.magnifyingglass")
                }
                .controlSize(.large)
                if !service.isModelLoaded() {
                    Label("Load a model first — the tokenizer must match the model that will use the document.",
                           systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                        .font(.callout)
                }
            }
        case .ready:
            readyForm
        case .tokenizing:
            progressRow("Tokenizing…")
        case .saving:
            progressRow("Saving tokens to disk…")
        case .done(let id):
            doneSummary(id)
        }
        if let err = errorMessage {
            Label(err, systemImage: "xmark.octagon.fill")
                .foregroundStyle(.red)
                .font(.callout)
        }
    }

    private var readyForm: some View {
        Form {
            LabeledContent("File") {
                Text(pickedURL?.lastPathComponent ?? "—")
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            LabeledContent("Source size") {
                Text(byteString(sourceByteCount))
                    .font(.system(.body, design: .monospaced))
            }
            TextField("Display name", text: $displayName)
        }
        .formStyle(.grouped)
    }

    private func progressRow(_ label: String) -> some View {
        HStack(spacing: 10) {
            ProgressView().controlSize(.small)
            Text(label)
                .foregroundStyle(.secondary)
        }
    }

    private func doneSummary(_ id: UUID) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Vectorized and saved.",
                   systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
            Text("\(tokenCount) tokens, \(byteString(sourceByteCount)) source.")
                .font(.callout)
                .foregroundStyle(.secondary)
            Text("Step 3 (precomputed KV cache) is not part of this build yet — the document is stored as a token sequence and will be turned into a prefill when the pipeline lands.")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }

    private var footer: some View {
        HStack {
            Spacer()
            switch phase {
            case .picking, .ready:
                Button("Cancel") { dismiss() }
                Button("Vectorize") { vectorize() }
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!canVectorize)
                    .buttonStyle(.borderedProminent)
            case .tokenizing, .saving:
                Button("Cancel") { dismiss() }
                    .disabled(true)
            case .done:
                Button("Done") { dismiss() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: [])
            }
        }
    }

    // MARK: - actions

    private var canVectorize: Bool {
        if case .ready = phase,
           pickedURL != nil,
           !displayName.trimmingCharacters(in: .whitespaces).isEmpty,
           service.isModelLoaded() {
            return true
        }
        return false
    }

    private func pickFile() {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.title = "Choose a text document"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        loadPickedFile(url: url)
        #endif
    }

    private func loadPickedFile(url: URL) {
        errorMessage = nil
        do {
            let data = try Data(contentsOf: url)
            // Try UTF-8; fall back to ISO-Latin-1 so logs / sources
            // from non-Unicode editors still come through. Binary
            // files come through as garbled text — the tokenizer
            // doesn't trap on weird input, it just produces a lot of
            // byte-level tokens, so we don't block here. The display
            // size + token count in the summary tell the user if
            // they picked a binary by mistake.
            let text = String(data: data, encoding: .utf8)
                ?? String(data: data, encoding: .isoLatin1)
                ?? ""
            self.pickedURL = url
            self.sourceText = text
            self.sourceByteCount = data.count
            self.displayName = url.deletingPathExtension().lastPathComponent
            self.phase = .ready
        } catch {
            self.errorMessage = "Read failed: \(error.localizedDescription)"
        }
    }

    private func vectorize() {
        guard let url = pickedURL,
              service.isModelLoaded() else {
            errorMessage = "Missing file or tokenizer."
            return
        }
        let trimmedName = displayName.trimmingCharacters(in: .whitespaces)
        guard !trimmedName.isEmpty else {
            errorMessage = "Name cannot be empty."
            return
        }
        let modelDir = service.currentModelDir()?.path ?? ""
        let fingerprint = ModelFingerprint.of(modelDirPath: modelDir)
        let text = sourceText
        let byteCount = sourceByteCount
        let filename = url.lastPathComponent

        phase = .tokenizing
        errorMessage = nil

        // BPE encode runs on the inference service's serial queue so
        // the main actor stays responsive and the non-Sendable
        // tokenizer never crosses an actor hop on its own.
        Task {
            guard let tokens = await service.tokenize(text) else {
                errorMessage = "Tokenizer unavailable."
                phase = .ready
                return
            }
            phase = .saving
            do {
                let doc = try library.add(
                    name: trimmedName,
                    sourceFilename: filename,
                    byteCount: byteCount,
                    tokens: tokens,
                    modelFingerprint: fingerprint)
                tokenCount = tokens.count
                phase = .done(doc.id)
            } catch {
                errorMessage = "Save failed: \(error.localizedDescription)"
                phase = .ready
            }
        }
    }

    // MARK: - helpers

    private func byteString(_ n: Int) -> String {
        if n < 1024 { return "\(n) B" }
        let kb = Double(n) / 1024.0
        if kb < 1024 { return String(format: "%.1f KB", kb) }
        let mb = kb / 1024.0
        return String(format: "%.1f MB", mb)
    }
}
