import XCTest
@testable import DS4Engine

/// HFTokenStore display redaction. Deliberately ONLY the pure helper: the
/// keychain-backed load/save/clear would touch the real user keychain from the
/// test process, so they are exercised manually through the Settings UI.
final class HFTokenStoreTests: XCTestCase {

    func testMaskedKeepsOnlyEdges() {
        let token = "hf_AbCdEfGhIjKlMnOpQrStUvWx"
        let masked = HFTokenStore.masked(token)
        XCTAssertEqual(masked, "hf_Ab…UvWx")
        XCTAssertFalse(masked.contains("EfGhIjKlMnOpQrSt"), "middle of the token must not leak")
    }

    func testMaskedShortTokensAreFullyHidden() {
        for short in ["", "x", "hf_short", "123456789012"] {
            XCTAssertEqual(HFTokenStore.masked(short), "•••", "'\(short)' must be fully hidden")
        }
    }
}
