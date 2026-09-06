import Foundation
import DS4Core
import DS4Metal

extension InferenceService {
    /// Local image chat. A nil history appends to the live conversation; a
    /// supplied history rebuilds its original text and image context.
    public func sendWithImages(userText: String, images: [ChatImage],
                               history: [VisionChatTurn]? = nil, systemPrompt: String? = nil,
                               thinkMode: DS4ThinkMode, sampling: SamplingParams,
                               maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task(priority: .userInitiated) {
                do {
                    guard let encoder = self.visionEncoder else {
                        throw VisionChatError.configuration("Carica il modello DeepSeek Vision Experimental e il suo encoder per usare immagini.")
                    }
                    let items = (history ?? []) + [VisionChatTurn(turn: .user(userText), images: images)]
                    var encoded: [(marker: String, embedding: DeepSeekV4VisionEmbedding)] = []
                    var turns: [ChatTurn] = []
                    for item in items {
                        try Task.checkCancellation()
                        guard item.images.count <= ChatImage.maximumImagesPerTurn else {
                            throw VisionChatError.invalidImage("Massimo 4 immagini per messaggio.")
                        }
                        guard case .user(let text) = item.turn else {
                            guard item.images.isEmpty else {
                                throw VisionChatError.invalidImage("Le immagini possono comparire solo nei messaggi utente.")
                            }
                            turns.append(item.turn)
                            continue
                        }
                        var markers = ""
                        for image in item.images {
                            guard !image.data.isEmpty, image.data.count <= ChatImage.maximumFileBytes else {
                                throw VisionChatError.invalidImage("\(image.name): immagine vuota o maggiore di 20 MB.")
                            }
                            continuation.yield(.progress("Analisi immagine: \(image.name)…"))
                            let embedding = try encoder.encode(imageData: image.data)
                            try Task.checkCancellation()
                            // A fresh, trusted marker keeps image framing out of
                            // untrusted user text and out of special-token parsing.
                            let marker = "DS4_IMAGE_" + UUID().uuidString + "_END"
                            encoded.append((marker, embedding))
                            markers += marker
                        }
                        turns.append(.user(markers + text))
                    }

                    if history != nil { self.resetConversation(systemPrompt: systemPrompt) }
                    let rendered: String
                    if history != nil {
                        let fullTurns = (self.systemPrompt.map { [ChatTurn.system($0)] } ?? []) + turns
                        rendered = ChatRenderer.render(turns: self.promptSafeTurns(fullTurns),
                            tools: self.promptSafeTools(self.tools), think: thinkMode.core,
                            markup: self.markup, compactTools: self.compactTools)
                    } else {
                        guard case .user(let text) = turns.last else {
                            throw VisionChatError.invalidImage("Messaggio immagine non valido.")
                        }
                        rendered = self.openingPrefix() + "<｜User｜>" + self.promptSafeText(text)
                            + self.assistantOpen(thinkMode)
                    }
                    var remaining = rendered[...]
                    var ids: [Int] = []
                    var blocks: [DeepSeekV4VisionPromptBlock] = []
                    for image in encoded {
                        guard let range = remaining.range(of: image.marker) else {
                            throw VisionChatError.invalidImage("Impossibile ricostruire la posizione dell’immagine nella chat.")
                        }
                        ids += self.tok.tokenizeRenderedChat(String(remaining[..<range.lowerBound])).map(Int.init)
                        let block = try encoder.promptBlock(embedding: image.embedding,
                            startPosition: self.committedIds.count + ids.count,
                            vocabularySize: self.tok.nVocab)
                        ids += block.tokens
                        blocks.append(block)
                        remaining = remaining[range.upperBound...]
                    }
                    ids += self.tok.tokenizeRenderedChat(String(remaining)).map(Int.init)
                    guard self.committedIds.count + ids.count < self.contextSize else {
                        throw InferenceError.contextExceeded(prompt: self.committedIds.count + ids.count,
                                                             context: self.contextSize)
                    }
                    self.visionBlocks += blocks
                    try self.generate(suffixIds: ids, think: thinkMode, sampling: sampling,
                                      maxTokens: maxTokens, continuation: continuation)
                    continuation.finish()
                } catch {
                    // Uncommitted images from a cancelled/failed prefill must
                    // never override a later suffix at the same positions.
                    self.visionBlocks.removeAll { $0.startPosition + $0.tokens.count > self.committedIds.count }
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    var visionEmbeddingOverrides: [Int: [Float]] {
        var values: [Int: [Float]] = [:]
        for block in visionBlocks {
            for (offset, row) in block.embeddings.enumerated() {
                values[block.startPosition + offset] = row
            }
        }
        return values
    }

    /// A visual span uses bidirectional attention. Its tokens must enter the
    /// decoder together, even when ordinary text prefill is chunked.
    func visionPrefillEnd(start: Int, proposedEnd: Int) -> Int {
        for block in visionBlocks {
            let blockEnd = block.startPosition + block.tokens.count
            if proposedEnd > block.startPosition && proposedEnd < blockEnd {
                return block.startPosition > start ? block.startPosition : blockEnd
            }
        }
        return proposedEnd
    }
}
