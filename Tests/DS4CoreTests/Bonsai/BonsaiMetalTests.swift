import XCTest
import Metal
@testable import DS4Core
@testable import DS4Metal

final class BonsaiMetalTests: XCTestCase {
    func testPackedKernelsRecurrentStateAndVisionCoordinates() throws {
        guard let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory else { throw XCTSkip("Requires Apple Silicon Metal") }
        // Compilation/PSO failure on a supported device is a test failure.
        let gpu = try BonsaiGPU(device: device)
        XCTAssertEqual(MemoryLayout<BonsaiArgs>.stride, 60)
        try checkHadamard(gpu)
        try checkPackedProjection(gpu, type: 142)
        try checkPackedProjection(gpu, type: 143)
        try checkConvolution(gpu)
        try checkRecurrentScan(gpu)
        try checkMRoPE(gpu)
    }

    private func floats(_ gpu: BonsaiGPU, _ values: [Float]) throws -> BonsaiBuffer {
        let view = try gpu.allocate(floats: values.count, label: "Bonsai test")
        values.withUnsafeBytes { view.buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return view
    }
    private func raw(_ gpu: BonsaiGPU, _ bytes: [UInt8]) throws -> BonsaiBuffer {
        let view = try gpu.allocate(floats: (bytes.count + 3) / 4, label: "Bonsai packed test")
        bytes.withUnsafeBytes { view.buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return view
    }
    private func read(_ view: BonsaiBuffer, _ count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: view.buffer.contents().advanced(by: view.offset).assumingMemoryBound(to: Float.self), count: count))
    }
    private func submit(_ gpu: BonsaiGPU, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        let cb = try XCTUnwrap(gpu.queue.makeCommandBuffer()), enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        do { try body(enc) } catch { enc.endEncoding(); throw error }
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed, cb.error?.localizedDescription ?? "")
        if cb.status != .completed { throw SwiftModelDecoderError.gpu("Bonsai synthetic command failed") }
    }
    private func exact(_ a: [Float], _ b: [Float], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern), file: file, line: line)
    }

    private func checkHadamard(_ gpu: BonsaiGPU) throws {
        let width = 6144, count = 2
        let input = (0..<width*count).map { Float(($0 * 17) % 97 - 48) / 64 }
        let signs = (0..<width).map { Int32($0 % 7 < 3 ? -1 : 1) }
        let x = try floats(gpu,input), y = try floats(gpu,[Float](repeating: -999, count: input.count))
        let signBytes = signs.withUnsafeBytes { Array($0) }, signBuffer = try raw(gpu,signBytes)
        try submit(gpu) { enc in
            try gpu.dispatch(enc,"hadamard",BonsaiArgs(n: width, heads: 48, dim: 128, groups: 16),[x,signBuffer,y],width/1024,count)
        }
        var expected: [Float] = []
        for row in 0..<count {
            let permuted = (0..<width).map { index -> Float in
                let head = index / 128, d = index % 128
                return input[row*width + ((head % 3) * 16 + head / 3)*128 + d]
            }
            expected += try BonsaiQuantization.hadamard(permuted,signs:signs)
        }
        exact(read(y,input.count),expected)
        // Embedding injection bypasses this inverse; ordinary lookup uses it.
        try submit(gpu) { enc in
            try gpu.dispatch(enc,"hadamard",BonsaiArgs(n: width, mode: 1),[y,signBuffer,x],width/1024,count)
        }
        for row in 0..<count {
            let restored = try BonsaiQuantization.hadamard(Array(expected[row*width..<(row+1)*width]),signs:signs,inverse:true)
            exact(read(x.row(row,width),width),restored)
        }
    }

    private func checkPackedProjection(_ gpu: BonsaiGPU, type: UInt32) throws {
        let columns = 1024, rows = 4113, count = 17
        let rowBytes = try XCTUnwrap(BonsaiQuantization.rowBytes(type:type,columns:columns))
        let blockBytes = type == 142 ? 34 : 28
        var packed = [UInt8](repeating:0,count:rowBytes*rows)
        for block in 0..<packed.count/blockBytes {
            for byte in 0..<blockBytes { packed[block*blockBytes+byte] = UInt8(truncatingIfNeeded:block*37+byte*61) }
            let scale = Float16(block % 3 == 0 ? -0.125 : 0.25).bitPattern
            let offset = block*blockBytes + (type == 142 ? 0 : 26)
            packed[offset] = UInt8(truncatingIfNeeded:scale); packed[offset+1] = UInt8(truncatingIfNeeded:scale >> 8)
        }
        let input = (0..<columns*count).map { Float(($0*29)%127 - 63) / 64 }
        let w = try raw(gpu,packed), x = try floats(gpu,input)
        let reference = try floats(gpu,[Float](repeating:-777,count:count*rows+16))
        let optimized = try floats(gpu,[Float](repeating:-777,count:count*rows+16))
        let a = BonsaiArgs(n:count,rows:rows,cols:columns,type:type,rowBytes:rowBytes)
        try submit(gpu) { enc in
            for token in 0..<count {
                try gpu.dispatch(enc,"mv",a,[w,x.row(token,columns),reference.row(token,rows)],(rows+3)/4,threads:128)
            }
            if type == 142 {
                try gpu.dispatch(enc,"mm_pq2_tiled",a,[w,x,optimized],(rows+63)/64,(count+31)/32,threads:128)
            } else {
                try gpu.dispatch(enc,"ptq_mm_8",a,[w,x,optimized],(rows+7)/8,(count+7)/8,threads:128)
            }
        }
        // Binary-fraction fixture removes reduction-order noise and exercises
        // row/token tails; canaries detect writes past the requested result.
        exact(read(optimized,count*rows+16),read(reference,count*rows+16))
        let expectedRow = try BonsaiQuantization.decodeRow(Array(packed.prefix(rowBytes)),type:type,columns:columns)
        let expected = zip(expectedRow,input.prefix(columns)).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        XCTAssertEqual(read(optimized,1)[0],expected,accuracy:1e-5)
        if type == 142 {
            try submit(gpu) { enc in
                try gpu.dispatch(enc,"pq2_mv",a,[w,x,optimized],(rows+15)/16,threads:64)
            }
            exact(read(optimized,rows),read(reference,rows))
        }
    }

    private func checkConvolution(_ gpu: BonsaiGPU) throws {
        let count = 17, channels = 128, width = 4
        let input = (0..<count*channels).map { Float(($0*31)%73-36)/128 }
        let coefficients = (0..<channels*width).map { Float(($0*13)%17-8)/16 }
        let x = try floats(gpu,input), w = try floats(gpu,coefficients)
        let h1 = try floats(gpu,[Float](repeating:0.01,count:channels*3)), h2 = try floats(gpu,read(h1,channels*3))
        let a = try floats(gpu,[Float](repeating:0,count:count*channels)), b = try floats(gpu,read(a,count*channels))
        try submit(gpu) { enc in
            try gpu.dispatch(enc,"conv_batch",BonsaiArgs(n:count,cols:channels,width:width),[x,w,h1,a],1)
            for row in 0..<count {
                try gpu.dispatch(enc,"conv",BonsaiArgs(n:channels,width:width),[x.row(row,channels),w,h2,b.row(row,channels)],1)
            }
        }
        exact(read(a,count*channels),read(b,count*channels)); exact(read(h1,channels*3),read(h2,channels*3))
    }

    private func checkRecurrentScan(_ gpu: BonsaiGPU) throws {
        let count = 17, dim = 128, heads = 6, keyHeads = 2, channels = (keyHeads*2+heads)*dim
        let qkv = (0..<count*channels).map { Float(($0*17)%71-35)/256 }
        let x = try floats(gpu,qkv)
        let alpha = try floats(gpu,(0..<count*heads).map { Float($0%9-4)/8 })
        let beta = try floats(gpu,(0..<count*heads).map { Float($0%7-3)/8 })
        let A = try floats(gpu,[Float](repeating:-0.25,count:heads)), dt = try floats(gpu,[Float](repeating:0.03,count:heads))
        let state = (0..<heads*dim*dim).map { Float($0%13-6)/1024 }
        let s1 = try floats(gpu,state), s2 = try floats(gpu,state)
        let y1 = try floats(gpu,[Float](repeating:0,count:count*heads*dim)), y2 = try floats(gpu,read(y1,count*heads*dim))
        let args = BonsaiArgs(n:count,heads:heads,kvheads:keyHeads,dim:dim)
        try submit(gpu) { enc in
            try gpu.dispatch(enc,"gdn_batch_128",args,[x,alpha,beta,A,dt,s1,y1],heads,threads:128)
            for row in 0..<count {
                try gpu.dispatch(enc,"gdn",args,[x.row(row,channels),alpha.row(row,heads),beta.row(row,heads),A,dt,s2,y2.row(row,heads*dim)],heads,threads:128)
            }
        }
        exact(read(y1,count*heads*dim),read(y2,count*heads*dim)); exact(read(s1,state.count),read(s2,state.count))
        // Continuing after a chunk uses the retained state, not a fresh scan.
        try submit(gpu) { enc in
            try gpu.dispatch(enc,"gdn_128",args,[x,alpha,beta,A,dt,s1,y1],heads,threads:128)
            try gpu.dispatch(enc,"gdn",args,[x,alpha,beta,A,dt,s2,y2],heads,threads:128)
        }
        exact(read(y1,heads*dim),read(y2,heads*dim)); exact(read(s1,state.count),read(s2,state.count))
    }

    private func checkMRoPE(_ gpu: BonsaiGPU) throws {
        let count = 3, heads = 2, dim = 256, rot = 64
        let positions: [Int32] = [9,2,5,10,3,5,11,3,6]
        let input = (0..<count*heads*dim).map { Float($0%53-26)/32 }
        let x = try floats(gpu,input)
        try submit(gpu) { enc in
            positions.withUnsafeBytes { enc.setBytes($0.baseAddress!,length:$0.count,index:2) }
            try gpu.dispatch(enc,"mrope",BonsaiArgs(n:count,heads:heads,dim:dim,rot:rot,base:10_000_000),[x],1,count)
        }
        let result = read(x,input.count)
        for row in 0..<count { for head in 0..<heads { for j in 0..<dim {
            let offset = (row*heads+head)*dim
            if j < rot/2 {
                let angle = Float(positions[row*3+j%3]) * pow(Float(10_000_000),-2*Float(j)/Float(rot))
                let p = input[offset+j], q = input[offset+j+rot/2]
                XCTAssertEqual(result[offset+j],p*cos(angle)-q*sin(angle),accuracy:1e-6)
                XCTAssertEqual(result[offset+j+rot/2],p*sin(angle)+q*cos(angle),accuracy:1e-6)
            } else if j >= rot { XCTAssertEqual(result[offset+j],input[offset+j]) }
        } } }
    }
}
