import Foundation

// Faithful Swift port of ds4_ssd.c — SSD streaming cache planning and argument
// parsing for routed-expert streaming. Pure logic, no engine dependencies.
//
// This is Phase 1 of the C -> Swift conversion. Behavior is matched 1:1 against
// the C originals (see DS4CoreTests/SSDCachePlanTests, which cross-checks every
// function against ds4_ssd.c through the CDS4 bridge). Where C relies on
// unsigned wraparound we use Swift's &*/&+ to reproduce it exactly rather than
// trap.

public enum SSDStreaming {
    /// 1 GiB, matching `DS4_GIB` in ds4_ssd.c.
    public static let gib: UInt64 = 1024 * 1024 * 1024

    /// A streaming cache budget expressed either as a byte amount or an explicit
    /// routed-expert count (the two forms `ds4_parse_streaming_cache_experts_arg`
    /// can return).
    public enum CacheSpec: Equatable, Sendable {
        case bytes(UInt64)
        case experts(UInt32)
    }

    /// Result of `autoCachePlan`, mirroring `ds4_ssd_cache_plan`.
    public struct CachePlan: Equatable, Sendable {
        public var modelTargetBytes: UInt64
        public var cacheBytes: UInt64
        public var effectiveCacheBytes: UInt64
        public var cacheExperts: UInt32
    }

    /// Port of `ds4_parse_gib_arg`. Accepts "<digits>" or "<digits>GB" (case
    /// insensitive on the suffix) and returns the value times 1 GiB. Returns nil
    /// for empty input, non-digits, zero, or values that would overflow.
    public static func parseGiBArg(_ s: String) -> UInt64? {
        let chars = Array(s.utf8)
        if chars.isEmpty { return nil }

        var len = chars.count
        // Strip a trailing "gb"/"GB" only when there is at least one digit before
        // it (C requires len > 2, so "GB" alone is rejected).
        if len > 2 {
            let g = chars[len - 2], b = chars[len - 1]
            if (g == UInt8(ascii: "g") || g == UInt8(ascii: "G")) &&
               (b == UInt8(ascii: "b") || b == UInt8(ascii: "B")) {
                len -= 2
            }
        }
        if len == 0 { return nil }

        var value: UInt64 = 0
        for i in 0..<len {
            let c = chars[i]
            guard c >= UInt8(ascii: "0"), c <= UInt8(ascii: "9") else { return nil }
            // value = value*10 + digit, faithful to strtoull (we already require
            // pure digits). Guard against overflow before it would trap.
            let digit = UInt64(c - UInt8(ascii: "0"))
            let (m, mo) = value.multipliedReportingOverflow(by: 10)
            if mo { return nil }
            let (a, ao) = m.addingReportingOverflow(digit)
            if ao { return nil }
            value = a
        }

        if value == 0 || value > UInt64.max / gib { return nil }
        return value * gib
    }

    /// Port of `ds4_parse_streaming_cache_experts_arg`. A "…GB" suffix means a
    /// byte budget; a bare integer means an explicit expert count.
    public static func parseCacheExpertsArg(_ s: String) -> CacheSpec? {
        let chars = Array(s.utf8)
        if chars.isEmpty { return nil }

        let len = chars.count
        if len > 2 {
            let g = chars[len - 2], b = chars[len - 1]
            if (g == UInt8(ascii: "g") || g == UInt8(ascii: "G")) &&
               (b == UInt8(ascii: "b") || b == UInt8(ascii: "B")) {
                guard let bytes = parseGiBArg(s) else { return nil }
                return .bytes(bytes)
            }
        }

        var value: UInt64 = 0
        for c in chars {
            guard c >= UInt8(ascii: "0"), c <= UInt8(ascii: "9") else { return nil }
            let digit = UInt64(c - UInt8(ascii: "0"))
            let (m, mo) = value.multipliedReportingOverflow(by: 10)
            if mo { return nil }
            let (a, ao) = m.addingReportingOverflow(digit)
            if ao { return nil }
            value = a
        }
        if value == 0 || value > UInt64(UInt32.max) { return nil }
        return .experts(UInt32(value))
    }

    /// Port of `ds4_ssd_cache_experts_for_byte_budget`.
    public static func cacheExpertsForByteBudget(bytes: UInt64, perExpertBytes: UInt64) -> UInt32 {
        if bytes == 0 || perExpertBytes == 0 { return 0 }
        let experts = bytes / perExpertBytes
        if experts == 0 || experts > UInt64(UInt32.max) { return 0 }
        return UInt32(experts)
    }

    /// Port of `ds4_ssd_auto_cache_plan`. Targets 4/5 of the recommended budget,
    /// subtracts the resident non-routed weights, and converts the remainder
    /// into a routed-expert count. Returns nil when inputs are degenerate.
    public static func autoCachePlan(recommendedBytes: UInt64,
                                     nonRoutedBytes: UInt64,
                                     perExpertBytes: UInt64,
                                     maxModelExperts: UInt64) -> CachePlan? {
        if recommendedBytes == 0 || perExpertBytes == 0 { return nil }

        let modelTarget = recommendedBytes > UInt64.max / 4
            ? UInt64.max
            : (recommendedBytes * 4) / 5

        var cacheBytes: UInt64 = 0
        if modelTarget > nonRoutedBytes {
            cacheBytes = modelTarget - nonRoutedBytes
        }

        var cacheExperts = cacheBytes / perExpertBytes
        if cacheExperts == 0 { cacheExperts = 1 }
        if maxModelExperts != 0 && cacheExperts > maxModelExperts {
            cacheExperts = maxModelExperts
        }
        if cacheExperts > UInt64(UInt32.max) { cacheExperts = UInt64(UInt32.max) }

        guard cacheExperts != 0 else { return nil }
        return CachePlan(modelTargetBytes: modelTarget,
                         cacheBytes: cacheBytes,
                         // C uses a plain (wrapping) multiply here.
                         effectiveCacheBytes: cacheExperts &* perExpertBytes,
                         cacheExperts: UInt32(cacheExperts))
    }
}
