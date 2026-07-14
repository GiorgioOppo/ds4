import XCTest
@testable import DS4Core

/// PromptLookup (draft n-gram per lo speculative decode): il matcher deve
/// preferire l'n-gramma più LUNGO, poi l'occorrenza più RECENTE, rispettare
/// minN e il clipping di `count`, e gestire testo periodico (match che si
/// sovrappone al suffisso).
final class PromptLookupTests: XCTestCase {

    func testNoRepetitionYieldsEmpty() {
        XCTAssertEqual(PromptLookup.draft(history: [1, 2, 3, 4, 5, 6], count: 3), [])
        XCTAssertEqual(PromptLookup.draft(history: [], count: 3), [])
        XCTAssertEqual(PromptLookup.draft(history: [7], count: 3), [])
    }

    func testSimpleRepeatCopiesContinuation() {
        // Suffisso [1,2,3] già visto all'inizio, seguito da 9.
        XCTAssertEqual(PromptLookup.draft(history: [1, 2, 3, 9, 8, 1, 2, 3], count: 2), [9, 8])
    }

    func testMostRecentOccurrenceWins() {
        // [5,6] ricorre due volte: la più RECENTE (indice 3) è seguita da 8.
        XCTAssertEqual(PromptLookup.draft(history: [5, 6, 7, 5, 6, 8, 5, 6], count: 1), [8])
    }

    func testLongerNgramBeatsShorter() {
        // Il 2-gramma [2,3] più recente è seguito da 9, ma il 3-gramma [1,2,3]
        // (match più lungo, quindi più affidabile) è seguito da 4.
        let h = [1, 2, 3, 4, 0, 2, 3, 9, 0, 1, 2, 3]
        XCTAssertEqual(PromptLookup.draft(history: h, count: 1), [4])
    }

    func testCountClippedByHistoryEnd() {
        // La continuazione dell'occorrenza è lunga 1 (il match è vicino alla
        // fine): chiede 4 candidati, ne ottiene quelli disponibili.
        XCTAssertEqual(PromptLookup.draft(history: [1, 2, 9, 1, 2], count: 4), [9, 1, 2])
    }

    func testPeriodicTextCopiesThePeriod() {
        // "4 4 4 4": il suffisso [4,4,4] matcha sovrapposto a i=0 → la
        // continuazione parte da indice 3 ed è clippata dalla fine: [4].
        XCTAssertEqual(PromptLookup.draft(history: [4, 4, 4, 4], count: 2), [4])
        // Periodo 2 senza clipping: il 4-gramma [7,8,7,8] matcha a i=0 e la
        // continuazione (da indice 4) prosegue il periodo: [7,8].
        XCTAssertEqual(PromptLookup.draft(history: [7, 8, 7, 8, 7, 8], count: 2), [7, 8])
    }

    func testMinNRespected() {
        // Solo un match di lunghezza 1 disponibile: sotto minN=2 → vuoto.
        XCTAssertEqual(PromptLookup.draft(history: [3, 1, 3, 2, 9, 3], count: 2), [])
    }

    func testCountZeroYieldsEmpty() {
        XCTAssertEqual(PromptLookup.draft(history: [1, 2, 1, 2], count: 0), [])
    }
}
