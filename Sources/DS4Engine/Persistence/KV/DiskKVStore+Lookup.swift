import Foundation
import DS4Core
import DS4Metal

extension DiskKVStore {
    // MARK: Lookup

    /// A matched entry: the token prefix plus WHERE it lives. The tensor body is
    /// NOT loaded here — restore streams it into the decoder one layer at a time.
    public struct Hit {
        public let tokens: [Int]
        let url: URL
    }

    /// Find the stored entry with the LONGEST token prefix of `ids` (strictly
    /// shorter than `ids`, ≥ minTokens, same model, fits this context). Cheap:
    /// scans headers + token lists only; the body is read by `restore`.
    public func findLongestPrefix(of ids: [Int], modelName: String) -> Hit? {
        var best: Hit?
        for url in entryURLs() {
            guard let scan = scanEntry(url) else { continue }
            // Local, NOT inline: `count < contextSize, … count > (…)` would be
            // parsed as a generic specialization `count<…>(…)` by Swift.
            let bestCount = best?.tokens.count ?? 0
            guard scan.model == modelName,
                  scan.tokens.count >= options.minTokens,
                  scan.tokens.count < ids.count,
                  scan.tokens.count < contextSize,
                  scan.tokens.count > bestCount,
                  ids.starts(with: scan.tokens) else { continue }
            best = Hit(tokens: scan.tokens, url: url)
        }
        return best
    }

    /// ALL stored entry lengths that are strict token-prefixes of `ids` (for the
    /// distributed restore negotiation: the coordinator intersects each worker's
    /// set and picks the longest length EVERY shard can restore). Sorted
    /// descending; header/token scans only, no tensor bodies.
    public func storedPrefixLengths(of ids: [Int], modelName: String) -> [Int] {
        var lengths: [Int] = []
        for url in entryURLs() {
            guard let scan = scanEntry(url) else { continue }
            guard scan.model == modelName,
                  scan.tokens.count >= options.minTokens,
                  scan.tokens.count < ids.count,
                  scan.tokens.count < contextSize,
                  ids.starts(with: scan.tokens) else { continue }
            lengths.append(scan.tokens.count)
        }
        return lengths.sorted(by: >)
    }

    /// Stream-restore `hit` into the decoder, ONE layer at a time: each layer's
    /// float slabs are parsed, written into the KV buffers, and released before
    /// the next layer is read — peak RAM is one layer, not the ~3× whole-file
    /// cost of the old load-everything path. Reads use F_NOCACHE (like store),
    /// so a multi-GB restore can't evict the hot page cache either. Bumps the
    /// hit counters on success; a corrupt entry is discarded like the C does.
    public func restore(_ hit: Hit, into decoder: StreamingDecoder) -> Bool {
        streamRestore(hit.url, into: decoder)
    }

    /// Restore the entry stored under the EXACT `tokens` (a content-keyed cache:
    /// the key is the file/project content prefix, not a conversation prefix).
    /// Same streaming path as `restore(_:into:)`. false if absent/corrupt/mismatched.
    public func restore(forTokens tokens: [Int], modelName: String,
                        into decoder: StreamingDecoder) -> Bool {
        let url = directory.appendingPathComponent(entryName(tokens: tokens, modelName: modelName))
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        return streamRestore(url, into: decoder)
    }

    private func streamRestore(_ url: URL, into decoder: StreamingDecoder) -> Bool {
        do {
            try streamSnapshot(url,
                onMeta: { nKeys, headDim, layerCount in
                    // Shape gate BEFORE any tensor body is read.
                    try decoder.beginImportKV(nKeys: nKeys, headDim: headDim, layerCount: layerCount)
                },
                onLayer: { i, layer in try decoder.importKVLayer(layer, at: i) })
            bumpHit(url)
            return true
        } catch is KVSnapshotError {
            return false            // valid entry, wrong shape for this decoder: keep it
        } catch {
            // Corrupt/truncated entry: discard it like the C does on load failure.
            try? FileManager.default.removeItem(at: url)
            return false
        }
    }

}
