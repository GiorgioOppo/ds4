import Foundation

/// DeepSeek's visual vocabulary is synthetic: these IDs must never index the
/// language embedding table. The matching sidecar supplies their embeddings.
public enum DeepSeekV4VisionToken: Int, Sendable {
    case start = 0, pad = 1, image = 2, newline = 3, end = 4
}

public struct DeepSeekV4VisionEmbedding: Sendable {
    public let gridHeight: Int
    public let gridWidth: Int
    /// Natural raster order, 4096 Float32 values per aligned image patch.
    public let values: [Float]

    public init(gridHeight: Int, gridWidth: Int, values: [Float]) {
        self.gridHeight = gridHeight
        self.gridWidth = gridWidth
        self.values = values
    }
}

public struct DeepSeekV4VisionPromptBlock: Sendable {
    public let startPosition: Int
    public let tokens: [Int]
    public let embeddings: [[Float]]

    public init(startPosition: Int, tokens: [Int], embeddings: [[Float]]) {
        self.startPosition = startPosition
        self.tokens = tokens
        self.embeddings = embeddings
    }
}

public enum DeepSeekV4VisionError: Error, LocalizedError, Sendable {
    case invalidModel(String)
    case invalidImage(String)
    case invalidLayout(String)

    public var errorDescription: String? {
        switch self {
        case .invalidModel(let detail): return "Encoder DeepSeek Vision non valido: \(detail)"
        case .invalidImage(let detail): return "Immagine non valida: \(detail)"
        case .invalidLayout(let detail): return "Sequenza DeepSeek Vision non valida: \(detail)"
        }
    }
}

/// Port of antirez/ds4's DeepSeek Vision-Exp packing. The start sentinel lands
/// at position 3 modulo 4; rows are interleaved in pairs for the KV compressor.
public struct DeepSeekV4VisionLayout: Sendable {
    public let types: [DeepSeekV4VisionToken]
    /// Source raster row for image entries in `types`, in encounter order.
    public let imagePermutation: [Int]

    public init(gridHeight: Int, gridWidth: Int, startPosition: Int) throws {
        guard (1...384).contains(gridHeight), (1...384).contains(gridWidth),
              startPosition >= 0 else {
            throw DeepSeekV4VisionError.invalidLayout("dimensioni o posizione fuori limite")
        }
        let rows = gridHeight + (gridHeight & 1)
        let rowLength = gridWidth + 1
        let compressionPadding = 3 - startPosition % 4
        let tailPadding = ((rows / 2 * rowLength) & 1) * 2
        let count = compressionPadding + 1 + rows * rowLength + tailPadding + 1
        guard count <= 384 else {
            throw DeepSeekV4VisionError.invalidLayout("il blocco supera 384 token")
        }
        var types = Array(repeating: DeepSeekV4VisionToken.pad, count: compressionPadding)
        var permutation: [Int] = []
        types.append(.start)
        for pair in 0..<(rows / 2) {
            for column in 0..<rowLength {
                for rowInPair in 0..<2 {
                    let row = pair * 2 + rowInPair
                    if row < gridHeight && column < gridWidth {
                        types.append(.image)
                        permutation.append(row * gridWidth + column)
                    } else {
                        types.append(row < gridHeight ? .newline : .pad)
                    }
                }
            }
        }
        types.append(contentsOf: repeatElement(.pad, count: tailPadding))
        types.append(.end)
        self.types = types
        self.imagePermutation = permutation
    }
}
