import Foundation
import DS4Core
import DS4Metal

extension DiskKVStore {
    // MARK: little-endian append helpers

    func appendU32(_ d: inout Data, _ v: UInt32) {
        d.append(contentsOf: [UInt8(v & 0xff), UInt8((v >> 8) & 0xff),
                              UInt8((v >> 16) & 0xff), UInt8((v >> 24) & 0xff)])
    }
}
