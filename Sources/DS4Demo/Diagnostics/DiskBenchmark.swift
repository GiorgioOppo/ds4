import Foundation
import Metal

// ── DS4_DIAG=1: diagnosi delle ottimizzazioni di streaming ──
// Stampa i numeri che decidono la PROSSIMA ottimizzazione: banda grezza del
// disco (il tetto del gather), presenza dei pesi MTP (decodifica speculativa
// possibile?), knob attivi, e — dopo la generazione — concentrazione del
// routing + allocazione slot per layer e il verdetto gather vs SSD.

/// Banda del file modello BYPASSANDO la page cache (F_NOCACHE) — misura il
/// disco vero, ripetibile. Tre scenari con slab da ~2 MB (la taglia di un
/// esperto 2-bit): sequenziale a coda 1, random a coda 1 (il gather senza
/// hint), random parallelo (il gather con madvise/pread paralleli). Il TETTO
/// del disco è il massimo dei tre: sugli NVMe Apple il parallelo random può
/// superare il sequenziale a coda 1 (serve profondità di coda, non località).
func diskBench(path: String) -> (report: String, ceilingGBs: Double) {
    let fd = open(path, O_RDONLY)
    guard fd >= 0 else { return ("  SSD: open fallita (\(String(cString: strerror(errno))))", 0) }
    defer { close(fd) }
    _ = fcntl(fd, F_NOCACHE, 1)
    var st = stat()
    guard fstat(fd, &st) == 0, st.st_size > 0 else { return ("  SSD: fstat fallita", 0) }
    let fileSize = Int(st.st_size)
    let slab = min(2 << 20, fileSize)            // ~2 MB
    let align = 1 << 14                          // offset a pagine da 16 KB
    let nSlabs = 24
    var rng = SystemRandomNumberGenerator()
    func randOff() -> Int {
        (Int.random(in: 0..<max(1, fileSize - slab), using: &rng) / align) * align
    }
    let buf = UnsafeMutableRawPointer.allocate(byteCount: slab, alignment: align)
    defer { buf.deallocate() }
    // 1) sequenziale (fino a 1 GB)
    let seqTotal = min(1 << 30, fileSize)
    var done = 0
    var t0 = Date()
    while done < seqTotal {
        let n = pread(fd, buf, min(slab, seqTotal - done), off_t(done))
        if n <= 0 { break }
        done += n
    }
    let seq = Double(done) / max(1e-9, Date().timeIntervalSince(t0)) / 1e9
    // 2) random, coda 1: uno slab alla volta
    let offs1 = (0..<nSlabs).map { _ in randOff() }
    t0 = Date()
    for off in offs1 { _ = pread(fd, buf, slab, off_t(off)) }
    let qd1 = Double(nSlabs * slab) / max(1e-9, Date().timeIntervalSince(t0)) / 1e9
    // 3) random, in parallelo: tutti gli slab insieme (pread è thread-safe)
    let offs2 = (0..<nSlabs).map { _ in randOff() }
    t0 = Date()
    DispatchQueue.concurrentPerform(iterations: offs2.count) { i in
        let b = UnsafeMutableRawPointer.allocate(byteCount: slab, alignment: align)
        defer { b.deallocate() }
        _ = pread(fd, b, slab, off_t(offs2[i]))
    }
    let qdN = Double(nSlabs * slab) / max(1e-9, Date().timeIntervalSince(t0)) / 1e9
    // 4) come il 3, ma DISALLINEATO come il gather VERO: il GGUF allinea i
    //    tensori a 32 byte e gli slab degli esperti hanno granularità di
    //    blocco quantizzato (66/84/144 B) — le pread del gather partono quasi
    //    sempre fuori pagina, e F_NOCACHE su I/O non allineato ricade (in
    //    parte) sul percorso bufferizzato. Se questa riga è molto sotto il
    //    "random parallelo", la leva è ALLINEARE le letture (finestra estesa
    //    alla pagina + fixup), non aggiungere thread.
    let offs3 = (0..<nSlabs).map { _ in randOff() + 66 }
    t0 = Date()
    DispatchQueue.concurrentPerform(iterations: offs3.count) { i in
        let b = UnsafeMutableRawPointer.allocate(byteCount: slab + 128, alignment: align)
        defer { b.deallocate() }
        _ = pread(fd, b + 66, slab, off_t(offs3[i]))
    }
    let qdU = Double(nSlabs * slab) / max(1e-9, Date().timeIntervalSince(t0)) / 1e9
    // 5) MTLIO (Metal fast resource loading, macOS 13+): stesso random
    //    parallelo, ma orchestrato dal runtime Metal dentro MTLBuffer di
    //    destinazione — l'equivalente Apple di DirectStorage e l'unico canale
    //    "più diretto" della pread F_NOCACHE (che già scrive in DMA dentro la
    //    memoria unificata). Se questa riga NON supera il "random parallelo",
    //    la leva DS4_MTLIO è chiusa per questo hardware.
    var mtlioLine = "    MTLIO               n/d"
    if let dev = MTLCreateSystemDefaultDevice() {
        do {
            let ioDesc = MTLIOCommandQueueDescriptor()
            ioDesc.type = .concurrent
            let ioq = try dev.makeIOCommandQueue(descriptor: ioDesc)
            let fh = try dev.makeIOFileHandle(url: URL(fileURLWithPath: path))
            let offsM = (0..<nSlabs).map { _ in randOff() }
            let bufs: [MTLBuffer] = offsM.map { _ in dev.makeBuffer(length: slab, options: .storageModeShared)! }
            // warm-up: la prima submission paga la creazione della coda IO
            if let wcb = ioq.makeCommandBuffer() as MTLIOCommandBuffer? {
                wcb.load(bufs[0], offset: 0, size: slab, sourceHandle: fh, sourceHandleOffset: 0)
                wcb.commit(); wcb.waitUntilCompleted()
            }
            let tM = Date()
            let iocb = ioq.makeCommandBuffer()
            for (i, off) in offsM.enumerated() {
                iocb.load(bufs[i], offset: 0, size: slab, sourceHandle: fh, sourceHandleOffset: off)
            }
            iocb.commit(); iocb.waitUntilCompleted()
            let gbs = Double(nSlabs * slab) / max(1e-9, Date().timeIntervalSince(tM)) / 1e9
            mtlioLine = String(format: "    MTLIO random        %6.2f GB/s   <- Metal fast resource loading (candidato DS4_MTLIO)", gbs)
        } catch {
            mtlioLine = "    MTLIO               errore: \(error.localizedDescription)"
        }
    }
    let ceiling = max(seq, qd1, qdN)
    let unalignedNote = qdU < qdN * 0.8
        ? "  <- PENALITÀ: allineare le pread del gather vale ~\(String(format: "%.1f", qdN / max(0.01, qdU)))×"
        : "  <- nessuna penalità significativa: l'allineamento NON è la leva"
    let report = String(format: """
      SSD (F_NOCACHE, slab %d MB):
        sequenziale coda 1  %6.2f GB/s
        random coda 1       %6.2f GB/s   <- gather senza hint/parallelismo
        random parallelo    %6.2f GB/s   <- gather con madvise/pread paralleli
        random DISALLINEATO %6.2f GB/s%@
        TETTO               %6.2f GB/s   <- riferimento per la banda effettiva
    """, slab >> 20, seq, qd1, qdN, qdU, unalignedNote, ceiling)
    return (report + "\n" + mtlioLine, ceiling)
}

