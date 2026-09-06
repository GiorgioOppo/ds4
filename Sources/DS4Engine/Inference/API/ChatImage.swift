import Foundation
import DS4Core

/// Encoded image bytes owned by the conversation. No file URL or network fetch
/// is needed when a saved chat is reopened.
public struct ChatImage: Codable, Sendable, Equatable {
    public var name: String
    public var data: Data

    public init(name: String, data: Data) {
        self.name = name
        self.data = data
    }

    public static let maximumFileBytes = 20 * 1_024 * 1_024
    public static let maximumImagesPerTurn = 4
}

/// A history turn with its original images, used when rebuilding a local vision
/// conversation. Text-only backends continue to use ChatTurn unchanged.
public struct VisionChatTurn: Sendable {
    public var turn: ChatTurn
    public var images: [ChatImage]

    public init(turn: ChatTurn, images: [ChatImage] = []) {
        self.turn = turn
        self.images = images
    }
}

public enum VisionChatError: Error, LocalizedError, CustomStringConvertible {
    case configuration(String)
    case invalidImage(String)

    public var description: String {
        switch self {
        case .configuration(let message), .invalidImage(let message): return message
        }
    }
    public var errorDescription: String? { description }
}
