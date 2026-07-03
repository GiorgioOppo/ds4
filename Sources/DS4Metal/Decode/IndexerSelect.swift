import Foundation

/// Top-K selection for the NSA indexer mask (StreamingDecoder.applyIndexerMask).
///
/// The selection order is the STRICT total order (score DESC, index ASC) — the
/// exact comparator the previous full sort used, so the selected SET is
/// bit-identical; only the cost drops from O(n log n) to O(n log k). That
/// matters because the selection runs per ratio-4 layer per token as soon as
/// the compressed rows exceed indexerTopK (~2k tokens of context): at 100k+
/// tokens a full sort of ~25k indices dominated the whole decode step.
/// Ties keep the LOWEST row index (the C argmax scan picks the first best).
enum IndexerSelect {
    /// allowed[i] == true for the k best of `count` scores; all false if k <= 0.
    static func allowedTopK(scores: UnsafePointer<Float>, count: Int, k: Int) -> [Bool] {
        var allowed = [Bool](repeating: false, count: max(0, count))
        guard count > 0, k > 0 else { return allowed }
        if k >= count {
            for i in 0..<count { allowed[i] = true }
            return allowed
        }
        // `better(a,b)`: a ranks strictly above b. A NaN score never ranks above
        // anything (the old sort comparator was undefined on NaN; this is safe).
        func better(_ a: Int, _ b: Int) -> Bool {
            scores[a] != scores[b] ? scores[a] > scores[b] : a < b
        }
        // Binary min-heap of the k best indices seen so far: the root is the
        // WORST kept element, replaced whenever a better candidate arrives.
        var heap = [Int](); heap.reserveCapacity(k)
        func siftDown(_ from: Int) {
            var i = from
            while true {
                let l = 2 * i + 1, r = 2 * i + 2
                var worst = i
                if l < heap.count, better(heap[worst], heap[l]) { worst = l }
                if r < heap.count, better(heap[worst], heap[r]) { worst = r }
                if worst == i { return }
                heap.swapAt(i, worst); i = worst
            }
        }
        func siftUp(_ from: Int) {
            var i = from
            while i > 0 {
                let p = (i - 1) / 2
                guard better(heap[p], heap[i]) else { return }
                heap.swapAt(i, p); i = p
            }
        }
        for i in 0..<count {
            if heap.count < k {
                heap.append(i); siftUp(heap.count - 1)
            } else if better(i, heap[0]) {
                heap[0] = i; siftDown(0)
            }
        }
        for i in heap { allowed[i] = true }
        return allowed
    }
}
