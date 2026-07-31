import Foundation
import Metal
import DS4Core

extension DenseStreamer {
    // MARK: Q8 compressor cache (<gguf>.q8comp.Lx-y)

    private static let compQ8Magic: UInt32 = 0x38435344  // "DSC8"
    private static let compQ8Version: UInt32 = 1

    static func loadQ8Compressors(
        rt: MetalRuntime, model: GGUFModel, fd: Int32,
        jobs: [(il: Int, f: Field, t: GGUFModel.Tensor)],
        skeleton: inout [Int: LayerWeights], lockResident: Bool
    ) throws {
        guard let lo = jobs.map(\.il).min(), let hi = jobs.map(\.il).max() else { return }
        let file = (model.path as NSString).lastPathComponent + ".q8comp.L\(lo)-\(hi)"
        let dir = ProcessInfo.processInfo.environment["DS4_Q4_CACHE_DIR"]
        let path: String
        if let dir, !dir.isEmpty {
            try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            path = dir + "/" + file
        } else {
            path = model.path + ".q8comp.L\(lo)-\(hi)"
        }
        let hashes = jobs.map { sourceHash(fd: fd, tensor: $0.t) }
        var tensors = loadQ8CompressorCache(rt: rt, path: path, modelSize: Int(model.size),
                                            jobs: jobs, hashes: hashes)
        if tensors == nil {
            LoadProgress.shared.begin("Quantizzazione Q8 compressori…", from: 0.30, to: 0.32,
                                      units: jobs.count)
            var out = [GPUTensor?](repeating: nil, count: jobs.count)
            let lock = NSLock()
            nonisolated(unsafe) var firstError: Error?
            nonisolated(unsafe) let rtRef = rt
            let js = jobs
            try out.withUnsafeMutableBufferPointer { buf in
                nonisolated(unsafe) let base = buf.baseAddress!
                DispatchQueue.concurrentPerform(iterations: jobs.count) { i in
                    do {
                        base[i] = try quantizeCompressorQ8(rt: rtRef, fd: fd, tensor: js[i].t)
                        LoadProgress.shared.advance()
                    } catch {
                        lock.lock(); if firstError == nil { firstError = error }; lock.unlock()
                    }
                }
                if let firstError { throw firstError }
            }
            tensors = try out.map {
                guard let t = $0 else { throw GGUFWeights.LoadError.message("Q8 compressor conversion failed") }
                return t
            }
            writeQ8CompressorCache(path: path, modelSize: Int(model.size), jobs: jobs,
                                   tensors: tensors!, hashes: hashes)
        }
        var bytes = 0
        for (i, job) in jobs.enumerated() {
            let t = tensors![i]
            if lockResident { t.lockResident() }
            switch job.f {
            case .compKv: skeleton[job.il]!.compKv = t; skeleton[job.il]!.compQ8 = true
            case .compGate: skeleton[job.il]!.compGate = t; skeleton[job.il]!.compQ8 = true
            case .idxKv: skeleton[job.il]!.idxKv = t; skeleton[job.il]!.idxCompQ8 = true
            case .idxGate: skeleton[job.il]!.idxGate = t; skeleton[job.il]!.idxCompQ8 = true
            default: break
            }
            bytes += t.byteLength
        }
        DS4Log.info("comp-q8", "\(jobs.count) proiezioni residenti, \(bytes >> 20) MB; cache: \(path)")
    }

    private static func quantizeCompressorQ8(rt: MetalRuntime, fd: Int32,
                                              tensor t: GGUFModel.Tensor) throws -> GPUTensor {
        guard let info = GGUF.typeInfo(t.type), info.name == "f16", Int(t.elements) % 32 == 0 else {
            throw GGUFWeights.LoadError.message("DS4_COMP_Q8 richiede compressori F16 multipli di 32: \(t.name)")
        }
        var src = [UInt8](repeating: 0, count: Int(t.bytes))
        guard src.withUnsafeMutableBytes({
            GGUFWeights.preadFull(fd, into: $0.baseAddress!, bytes: $0.count, offset: Int(t.absOffset))
        }) else { throw GGUFWeights.LoadError.message("pread Q8 compressor fallita: \(t.name)") }
        let bytes = Int(t.elements) / 32 * 34
        let out = try GPUTensor.uninitializedBytes(rt, byteLength: bytes, elementCount: Int(t.elements))
        src.withUnsafeBytes {
            Quantize.quantizeF16Q8_0($0.baseAddress!, count: Int(t.elements), into: out.buffer.contents())
        }
        return out
    }

    private static func loadQ8CompressorCache(
        rt: MetalRuntime, path: String, modelSize: Int,
        jobs: [(il: Int, f: Field, t: GGUFModel.Tensor)], hashes: [UInt64]
    ) -> [GPUTensor]? {
        let cfd = open(path, O_RDONLY)
        guard cfd >= 0 else { return nil }
        defer { close(cfd) }
        _ = fcntl(cfd, F_NOCACHE, 1)
        let headerBytes = 20 + jobs.count * 32
        var head = [UInt8](repeating: 0, count: headerBytes)
        guard head.withUnsafeMutableBytes({ GGUFWeights.preadFull(cfd, into: $0.baseAddress!, bytes: headerBytes, offset: 0) }) else { return nil }
        func u32(_ o: Int) -> UInt32 { head.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) } }
        func u64(_ o: Int) -> UInt64 { head.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt64.self) } }
        guard u32(0) == compQ8Magic, u32(4) == compQ8Version,
              u64(8) == UInt64(modelSize), Int(u32(16)) == jobs.count else { return nil }
        var out: [GPUTensor] = []
        out.reserveCapacity(jobs.count)
        for i in jobs.indices {
            let o = 20 + i * 32
            let expected = Int(jobs[i].t.elements) / 32 * 34
            guard Int(u32(o)) == jobs[i].il, Int(u32(o + 4)) == jobs[i].f.rawValue,
                  Int(u64(o + 8)) == expected, u64(o + 16) == hashes[i] else { return nil }
            guard let t = try? GPUTensor.uninitializedBytes(rt, byteLength: expected,
                                                            elementCount: Int(jobs[i].t.elements)),
                  GGUFWeights.preadFull(cfd, into: t.buffer.contents(), bytes: expected,
                                        offset: Int(u64(o + 24))) else { return nil }
            out.append(t)
        }
        return out
    }

    private static func writeQ8CompressorCache(
        path: String, modelSize: Int, jobs: [(il: Int, f: Field, t: GGUFModel.Tensor)],
        tensors: [GPUTensor], hashes: [UInt64]
    ) {
        let align = 4096
        var offset = (20 + jobs.count * 32 + align - 1) / align * align
        var head = Data()
        func p32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { head.append(contentsOf: $0) } }
        func p64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { head.append(contentsOf: $0) } }
        p32(compQ8Magic); p32(compQ8Version); p64(UInt64(modelSize)); p32(UInt32(jobs.count))
        var offsets: [Int] = []
        for i in jobs.indices {
            p32(UInt32(jobs[i].il)); p32(UInt32(jobs[i].f.rawValue)); p64(UInt64(tensors[i].byteLength))
            p64(hashes[i]); p64(UInt64(offset)); offsets.append(offset)
            offset += (tensors[i].byteLength + align - 1) / align * align
        }
        let tmp = path + ".tmp"
        try? FileManager.default.removeItem(atPath: tmp)
        guard FileManager.default.createFile(atPath: tmp, contents: nil), let fh = FileHandle(forWritingAtPath: tmp) else { return }
        do {
            try fh.write(contentsOf: head)
            for i in tensors.indices {
                try fh.seek(toOffset: UInt64(offsets[i]))
                try fh.write(contentsOf: Data(bytesNoCopy: tensors[i].buffer.contents(), count: tensors[i].byteLength, deallocator: .none))
            }
            try fh.close(); try? FileManager.default.removeItem(atPath: path)
            try FileManager.default.moveItem(atPath: tmp, toPath: path)
        } catch { try? fh.close(); try? FileManager.default.removeItem(atPath: tmp) }
    }

}
