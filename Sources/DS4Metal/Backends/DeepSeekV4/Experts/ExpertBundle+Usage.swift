import Foundation
import Metal
import DS4Core

extension ExpertBundle {
    func noteUse() {
        useLock.lock()
        served += 1
        let n = served
        var beat = false
        if n >= nextBeat { beat = true; nextBeat *= 2 }
        useLock.unlock()
        if n == 1 { Self.log("in uso: primo esperto servito dal sidecar") }
        else if beat { Self.log("in uso: \(n) esperti serviti dal sidecar") }
    }
}
