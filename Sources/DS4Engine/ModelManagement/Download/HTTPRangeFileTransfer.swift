import Foundation
import CryptoKit

struct HTTPRangeTransferResult: Sendable {
    let byteCount: Int64
    let totalBytes: Int64?
    let computedSHA256: String?
}

/// URLSession delegate that writes Foundation-provided `Data` blocks directly
/// to disk. This avoids one Swift async iteration per byte, which is essential
/// for GGUFs measured in hundreds of gigabytes.
final class HTTPRangeFileTransfer: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    /// `OperationQueue.underlyingQueue` is unowned, so retain this queue for the
    /// process lifetime. Sharing it also prevents two independent downloads
    /// from processing large body callbacks at the same instant.
    private static let delegateDispatchQueue = DispatchQueue(
        label: "org.ds4.model-download.delegate",
        qos: .utility,
        autoreleaseFrequency: .workItem
    )

    private let request: URLRequest
    private let partialURL: URL
    private let metadataURL: URL
    private let requestedOffset: Int64
    private let expectedSizeBytes: Int64?
    private let expectedSHA256: String?
    private let onProgress: @Sendable (ModelDownloadProgress) -> Void
    private let onState: @Sendable (ModelDownloadState) -> Void

    private let completionLock = NSLock()
    private var continuation: CheckedContinuation<HTTPRangeTransferResult, Error>?
    private var session: URLSession?
    private var task: URLSessionDataTask?
    private var cancelled = false
    private var completed = false

    // The delegate queue is serial, so streaming fields need no separate lock.
    private var fileHandle: FileHandle?
    private var streamError: Error?
    private var acceptedResponse = false
    private var done: Int64 = 0
    private var total: Int64?
    private var validator: String?
    private var hasher: SHA256?
    private var lastReportedBytes: Int64 = 0
    private var lastReportedTime = Date.timeIntervalSinceReferenceDate

    init(
        request: URLRequest,
        partialURL: URL,
        metadataURL: URL,
        requestedOffset: Int64,
        expectedSizeBytes: Int64?,
        expectedSHA256: String?,
        onProgress: @escaping @Sendable (ModelDownloadProgress) -> Void,
        onState: @escaping @Sendable (ModelDownloadState) -> Void
    ) {
        self.request = request
        self.partialURL = partialURL
        self.metadataURL = metadataURL
        self.requestedOffset = requestedOffset
        self.expectedSizeBytes = expectedSizeBytes
        self.expectedSHA256 = expectedSHA256
        self.onProgress = onProgress
        self.onState = onState
    }

    func start() async throws -> HTTPRangeTransferResult {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                begin(continuation: continuation)
            }
        } onCancel: {
            cancel()
        }
    }

    func cancel() {
        let taskToCancel: URLSessionDataTask? = completionLock.withLock {
            cancelled = true
            return task
        }
        taskToCancel?.cancel()
    }

    private func begin(
        continuation: CheckedContinuation<HTTPRangeTransferResult, Error>
    ) {
        let wasCancelled = completionLock.withLock { () -> Bool in
            self.continuation = continuation
            return cancelled
        }
        guard !wasCancelled else {
            complete(.failure(CancellationError()))
            return
        }

        let configuration = Self.makeSessionConfiguration()
        let queue = Self.makeDelegateQueue()

        let session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
        let task = session.dataTask(with: request)
        completionLock.withLock {
            self.session = session
            self.task = task
        }
        if completionLock.withLock({ cancelled }) {
            task.cancel()
        } else {
            task.resume()
        }
    }

    /// Keep the transport itself memory-bounded. Ephemeral storage alone can
    /// still create in-memory cookie/credential stores, so disable them
    /// explicitly as GGUF requests only need the request's bearer header.
    static func makeSessionConfiguration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.urlCredentialStorage = nil
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 60 * 60 * 24 * 30
        configuration.waitsForConnectivity = true
        configuration.httpMaximumConnectionsPerHost = 1
        return configuration
    }

    /// A serial delegate queue guarantees that URLSession cannot pile up
    /// multiple GGUF body blocks while the previous one is being written.
    /// Per-work-item autorelease pools promptly drain Foundation bridge objects.
    static func makeDelegateQueue() -> OperationQueue {
        let queue = OperationQueue()
        queue.name = "org.ds4.model-download"
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .utility
        queue.underlyingQueue = delegateDispatchQueue
        return queue
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse else {
            reject(ModelDownloader.DownloadError.http(-1), completionHandler: completionHandler)
            return
        }

        if (http.statusCode == 200 || http.statusCode == 206),
           let encoding = http.value(forHTTPHeaderField: "Content-Encoding"),
           !encoding.isEmpty,
           encoding.caseInsensitiveCompare("identity") != .orderedSame {
            reject(
                ModelDownloader.DownloadError.unexpectedContentEncoding(encoding),
                completionHandler: completionHandler
            )
            return
        }

        let etag = http.value(forHTTPHeaderField: "ETag")
        let candidateValidator = etag.flatMap { $0.hasPrefix("W/") ? nil : $0 }
            ?? http.value(forHTTPHeaderField: "Last-Modified")
        validator = candidateValidator.flatMap {
            $0.utf8.count <= ResumeMetadata.maximumValidatorBytes ? $0 : nil
        }

        do {
            switch http.statusCode {
            case 200:
                // A server may ignore Range or reject an old If-Range validator.
                // Restart safely: open and TRUNCATE rather than seek to the end.
                let responseTotal = positiveContentLength(http)
                try validateExpectedSize(responseTotal)
                // Truncating an old `.part` releases `requestedOffset` bytes on
                // this same volume. Count that reclaimable space; otherwise a
                // Range-ignoring server can produce a false "disk full" error
                // even though restarting the download fits after truncation.
                try validateCapacity(
                    totalBytes: responseTotal,
                    existingBytes: requestedOffset
                )
                try openPartial(at: 0, truncate: true)
                done = 0
                total = responseTotal
                if expectedSHA256 != nil { hasher = SHA256() }
                persistResumeMetadata()
                acceptedResponse = true
                onState(.downloading)
                onProgress(.init(completedBytes: 0, totalBytes: total))
                completionHandler(.allow)

            case 206:
                guard let parsed = ContentRange.satisfied(
                    http.value(forHTTPHeaderField: "Content-Range")
                ), parsed.start == requestedOffset else {
                    throw ModelDownloader.DownloadError.invalidContentRange(
                        http.value(forHTTPHeaderField: "Content-Range")
                    )
                }
                if let stored = ResumeMetadata.load(from: metadataURL)?.validator,
                   let validator, stored != validator {
                    throw ModelDownloader.DownloadError.remoteObjectChanged
                }
                try validateExpectedSize(parsed.total)
                try validateCapacity(totalBytes: parsed.total, existingBytes: requestedOffset)
                try openPartial(at: requestedOffset, truncate: false)
                done = requestedOffset
                total = parsed.total
                // A resumed hash is verified by one bounded-memory disk pass after
                // transfer. Seeding SHA here would still require reading the part.
                hasher = nil
                persistResumeMetadata()
                acceptedResponse = true
                if requestedOffset > 0 { onState(.resuming(fromByte: requestedOffset)) }
                onState(.downloading)
                onProgress(.init(completedBytes: done, totalBytes: total))
                completionHandler(.allow)

            case 416:
                let remoteTotal = ContentRange.unsatisfiedTotal(
                    http.value(forHTTPHeaderField: "Content-Range")
                )
                guard requestedOffset > 0,
                      let remoteTotal,
                      remoteTotal == requestedOffset else {
                    throw ModelDownloader.DownloadError.rangeNotSatisfiable(
                        partialBytes: requestedOffset,
                        remoteBytes: remoteTotal
                    )
                }
                try validateExpectedSize(remoteTotal)
                done = requestedOffset
                total = remoteTotal
                acceptedResponse = true
                onState(.resuming(fromByte: requestedOffset))
                onProgress(.init(completedBytes: done, totalBytes: total))
                completionHandler(.allow)

            default:
                throw ModelDownloader.DownloadError.http(http.statusCode)
            }
        } catch {
            reject(error, completionHandler: completionHandler)
        }
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive data: Data
    ) {
        guard streamError == nil, !data.isEmpty else { return }
        do {
            guard let fileHandle else {
                throw ModelDownloader.DownloadError.http(-1)
            }
            try fileHandle.write(contentsOf: data)
            hasher?.update(data: data)
            done += Int64(data.count)
            if let total, done > total {
                throw ModelDownloader.DownloadError.incompleteDownload(
                    expected: total,
                    actual: done
                )
            }
            reportProgressIfNeeded()
        } catch {
            streamError = error
            dataTask.cancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        var redirected = request
        // The first request may contain a private HF token. Redirect targets use
        // signed CDN URLs, so the bearer credential must never cross host names.
        if redirected.url?.scheme?.lowercased() != "https"
            || redirected.url?.host?.lowercased() != "huggingface.co" {
            redirected.setValue(nil, forHTTPHeaderField: "Authorization")
        }
        completionHandler(redirected)
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        do {
            try fileHandle?.close()
        } catch {
            streamError = streamError ?? error
        }
        fileHandle = nil

        if let streamError {
            complete(.failure(streamError))
            return
        }
        if let error {
            let code = (error as NSError).code
            if completionLock.withLock({ cancelled }) || code == NSURLErrorCancelled {
                complete(.failure(CancellationError()))
            } else {
                complete(.failure(error))
            }
            return
        }
        guard acceptedResponse else {
            complete(.failure(ModelDownloader.DownloadError.http(-1)))
            return
        }
        if let total, done != total {
            complete(.failure(ModelDownloader.DownloadError.incompleteDownload(
                expected: total,
                actual: done
            )))
            return
        }

        onProgress(.init(completedBytes: done, totalBytes: total))
        let digest = hasher.map { ModelDownloader.hex($0.finalize()) }
        complete(.success(.init(
            byteCount: done,
            totalBytes: total,
            computedSHA256: digest
        )))
    }

    private func reject(
        _ error: Error,
        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
    ) {
        streamError = error
        completionHandler(.cancel)
    }

    private func complete(_ result: Result<HTTPRangeTransferResult, Error>) {
        let continuation: CheckedContinuation<HTTPRangeTransferResult, Error>? =
            completionLock.withLock {
                guard !completed else { return nil }
                completed = true
                let value = self.continuation
                self.continuation = nil
                task = nil
                return value
            }
        guard let continuation else { return }
        session?.finishTasksAndInvalidate()
        session = nil
        switch result {
        case .success(let value): continuation.resume(returning: value)
        case .failure(let error): continuation.resume(throwing: error)
        }
    }

    private func openPartial(at offset: Int64, truncate: Bool) throws {
        if (try? FileManager.default.destinationOfSymbolicLink(atPath: partialURL.path)) != nil {
            throw ModelDownloader.DownloadError.partialItemNotRegular(partialURL.path)
        }
        if !FileManager.default.fileExists(atPath: partialURL.path) {
            guard FileManager.default.createFile(atPath: partialURL.path, contents: nil) else {
                throw CocoaError(.fileWriteUnknown)
            }
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: partialURL.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular else {
            throw ModelDownloader.DownloadError.partialItemNotRegular(partialURL.path)
        }
        let localSize = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        if !truncate, localSize != offset {
            throw ModelDownloader.DownloadError.invalidContentRange(
                "local .part changed from \(offset) to \(localSize) bytes"
            )
        }
        let handle = try FileHandle(forWritingTo: partialURL)
        _ = ModelDownloader.enableUncachedIO(for: handle)
        if truncate {
            try handle.truncate(atOffset: 0)
            try handle.seek(toOffset: 0)
        } else {
            try handle.seek(toOffset: UInt64(offset))
        }
        fileHandle = handle
        lastReportedBytes = offset
        lastReportedTime = Date.timeIntervalSinceReferenceDate
    }

    private func validateExpectedSize(_ remoteSize: Int64?) throws {
        guard let expectedSizeBytes, let remoteSize, expectedSizeBytes != remoteSize else { return }
        throw ModelDownloader.DownloadError.existingFileSizeMismatch(
            expected: expectedSizeBytes,
            actual: remoteSize
        )
    }

    private func validateCapacity(totalBytes: Int64?, existingBytes: Int64) throws {
        guard let totalBytes,
              let available = ModelDownloader.availableCapacity(
                at: partialURL.deletingLastPathComponent()
              ) else { return }
        let required = ModelDownloader.requiredFreeSpace(
            totalBytes: totalBytes,
            existingPartialBytes: existingBytes
        )
        guard available >= required else {
            throw ModelDownloader.DownloadError.insufficientDiskSpace(
                required: required,
                available: available
            )
        }
    }

    private func persistResumeMetadata() {
        ResumeMetadata(validator: validator, totalBytes: total).save(to: metadataURL)
    }

    private func reportProgressIfNeeded() {
        let now = Date.timeIntervalSinceReferenceDate
        let enoughBytes = done - lastReportedBytes >= 64 << 20
        let enoughTime = now - lastReportedTime >= 0.2
        guard enoughBytes && enoughTime else { return }
        lastReportedBytes = done
        lastReportedTime = now
        onProgress(.init(completedBytes: done, totalBytes: total))
    }

    private func positiveContentLength(_ response: HTTPURLResponse) -> Int64? {
        if response.expectedContentLength > 0 { return response.expectedContentLength }
        guard let raw = response.value(forHTTPHeaderField: "Content-Length"),
              let value = Int64(raw), value > 0 else { return nil }
        return value
    }
}

private struct ContentRange {
    let start: Int64
    let end: Int64
    let total: Int64

    static func satisfied(_ value: String?) -> ContentRange? {
        guard let value else { return nil }
        let pieces = value.split(separator: " ", maxSplits: 1)
        guard pieces.count == 2, pieces[0].lowercased() == "bytes" else { return nil }
        let rangeAndTotal = pieces[1].split(separator: "/", maxSplits: 1)
        guard rangeAndTotal.count == 2,
              let total = Int64(rangeAndTotal[1]), total > 0 else { return nil }
        let bounds = rangeAndTotal[0].split(separator: "-", maxSplits: 1)
        guard bounds.count == 2,
              let start = Int64(bounds[0]),
              let end = Int64(bounds[1]),
              start >= 0, end >= start, end < total else { return nil }
        return .init(start: start, end: end, total: total)
    }

    static func unsatisfiedTotal(_ value: String?) -> Int64? {
        guard let value else { return nil }
        let pieces = value.split(separator: " ", maxSplits: 1)
        guard pieces.count == 2, pieces[0].lowercased() == "bytes" else { return nil }
        let rangeAndTotal = pieces[1].split(separator: "/", maxSplits: 1)
        guard rangeAndTotal.count == 2, rangeAndTotal[0] == "*",
              let total = Int64(rangeAndTotal[1]), total >= 0 else { return nil }
        return total
    }
}
