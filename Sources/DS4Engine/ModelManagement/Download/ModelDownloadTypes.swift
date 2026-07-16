import Foundation

public enum ModelDownloadDisposition: String, Sendable, Hashable {
    case alreadyPresent
    case downloaded
}

public struct ModelDownloadResult: Sendable, Hashable {
    public let disposition: ModelDownloadDisposition
    public let fileURL: URL
    public let byteCount: Int64

    public init(disposition: ModelDownloadDisposition, fileURL: URL, byteCount: Int64) {
        self.disposition = disposition
        self.fileURL = fileURL
        self.byteCount = byteCount
    }
}

public enum ModelDownloadState: Sendable, Hashable {
    case checkingLocalFile
    case preparingRequest
    case resuming(fromByte: Int64)
    case downloading
    case verifying
    case finalizing
    case completed(ModelDownloadDisposition)
}

public struct ModelDownloadProgress: Sendable, Hashable {
    public let completedBytes: Int64
    public let totalBytes: Int64?

    public init(completedBytes: Int64, totalBytes: Int64?) {
        self.completedBytes = max(0, completedBytes)
        self.totalBytes = totalBytes.flatMap { $0 > 0 ? $0 : nil }
    }

    public var fractionCompleted: Double? {
        guard let totalBytes, totalBytes > 0 else { return nil }
        return min(1, max(0, Double(completedBytes) / Double(totalBytes)))
    }
}
