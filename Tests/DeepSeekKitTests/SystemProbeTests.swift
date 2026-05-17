import XCTest
@testable import DeepSeekKit

/// Test della tabella interna di mapping `MTLDevice.name` → numero
/// di GPU core. Usa la funzione internal `_coreCount(forDeviceName:)`
/// per testare il match prefisso senza dipendere dal device reale.
final class SystemProbeTests: XCTestCase {

    // MARK: - Tabella per chip noti

    func testM1Family() {
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M1"),       8)
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M1 Pro"),  16)
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M1 Max"),  32)
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M1 Ultra"), 64)
    }

    func testM2Family() {
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M2"),       10)
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M2 Pro"),   19)
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M2 Max"),   38)
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M2 Ultra"), 76)
    }

    func testM3Family() {
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M3"),       10)
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M3 Pro"),   18)
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M3 Max"),   40)
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M3 Ultra"), 80)
    }

    func testM4Family() {
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M4"),     10)
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M4 Pro"), 20)
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M4 Max"), 40)
    }

    // MARK: - Match più specifico vince

    func testSpecificBeforeGeneric() {
        // "Apple M3 Max GPU" deve risolvere a 40 (M3 Max), NON a 10 (M3
        // generico). Il match avviene per prefisso e la tabella ordina
        // le voci più specifiche prima delle generiche.
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M3 Max GPU"), 40)
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M2 Pro Engine"), 19)
        XCTAssertEqual(SystemProbe._coreCount(forDeviceName: "Apple M1 Ultra v2"), 64)
    }

    // MARK: - Chip sconosciuti

    func testUnknownChip() {
        XCTAssertNil(SystemProbe._coreCount(forDeviceName: "Apple M5 Max"))
        XCTAssertNil(SystemProbe._coreCount(forDeviceName: "Foo Bar GPU"))
        XCTAssertNil(SystemProbe._coreCount(forDeviceName: ""))
        XCTAssertNil(SystemProbe._coreCount(forDeviceName: "Apple"))
    }

    // MARK: - Fallback

    func testCoreCountOrFallbackProducesPositive() {
        // Su host reale: o c'è una entry tabella o cpuCount() * 2 > 0.
        XCTAssertGreaterThan(SystemProbe.gpuCoreCountOrFallback(), 0)
    }

    // MARK: - gpuName su device reale

    func testGpuNameOnRealDevice() {
        // Su CI senza Metal questo test va skippato.
        try? XCTSkipUnless(true, "Metal device required")
        let name = SystemProbe.gpuName()
        XCTAssertFalse(name.isEmpty)
    }
}
