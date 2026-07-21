import Foundation

/// IQ2_XXS ENCODER — port of `ds4q_write_iq2_xxs_block` and its lattice
/// tables from ds4's `gguf-tools/quants.c` (MIT, derived from GGML).
///
/// A 256-value block is quantized as eight 32-value groups; each group
/// stores four 8-value grid indices plus four 7-bit sign masks, and a
/// single f16 block scale refined by 4-bit per-group scale nibbles. Not
/// every 2-bit 8-tuple is a valid lattice point: init builds the direct
/// map for allowed tuples and a nearest-neighbour list for the missing
/// ones, matching the GGML search exactly.
extension QuantEncode {
    struct IQ2XXSData: @unchecked Sendable {
        let grid: [[Int8]]        // 256 × 8 lattice points (values 1,3,5,7)
        let map: [Int32]          // code -> grid index, or -(offset+1) into neighbours
        let neighbours: [UInt16]  // [count, idx...] runs, offset-addressed
    }

    /// Lazy and thread-safe by Swift static-let semantics (the C uses a
    /// mutex for the same one-time build).
    static let iq2xxs: IQ2XXSData = buildIQ2XXS()

    private static let iq2xxsGridCodes: [UInt16] = [
            0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
          100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
         1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
         1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
         2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
         4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
         5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
         8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
        10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
        16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
        17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
        20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
        22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
        25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
        33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
        37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
    ]

    private static func buildIQ2XXS() -> IQ2XXSData {
        let gridSize = 256
        let mapSize = 43692
        let neighbourShells = 2
        let codes = iq2xxsGridCodes

        var grid = [[Int8]](repeating: [Int8](repeating: 0, count: 8),
                            count: gridSize)
        for k in 0..<gridSize {
            for i in 0..<8 {
                let l = (Int(codes[k]) >> (2 * i)) & 3
                grid[k][i] = Int8(2 * l + 1)
            }
        }

        var map = [Int32](repeating: -1, count: mapSize)
        for i in 0..<gridSize { map[Int(codes[i])] = Int32(i) }

        var neighbours: [UInt16] = []
        var dist2 = [(d2: Int, index: Int)](repeating: (0, 0), count: gridSize)
        var counter = 0
        for i in 0..<mapSize where map[i] < 0 {
            var pos = [Int8](repeating: 0, count: 8)
            for k in 0..<8 { pos[k] = Int8(2 * ((i >> (2 * k)) & 3) + 1) }
            for j in 0..<gridSize {
                var d2 = 0
                for k in 0..<8 {
                    let diff = Int(grid[j][k]) - Int(pos[k])
                    d2 += diff * diff
                }
                dist2[j] = (d2, j)
            }
            dist2.sort { $0.d2 != $1.d2 ? $0.d2 < $1.d2 : $0.index < $1.index }
            map[i] = Int32(-(counter + 1))
            var shell = dist2[0].d2
            var have = 1
            let startIndex = neighbours.count
            neighbours.append(0)
            counter += 1
            var n = 0
            for j in 0..<gridSize {
                if dist2[j].d2 > shell {
                    if have == neighbourShells { break }
                    shell = dist2[j].d2
                    have += 1
                }
                neighbours.append(UInt16(dist2[j].index))
                counter += 1
                n += 1
            }
            neighbours[startIndex] = UInt16(n)
        }
        return IQ2XXSData(grid: grid, map: map, neighbours: neighbours)
    }

    /// Port of `ds4q_iq2_find_best_neighbour`: weighted nearest lattice
    /// point among the precomputed shells; writes the 2-bit levels into L.
    @discardableResult
    private static func findBestNeighbour(offset: Int,
                                          data: IQ2XXSData,
                                          xval: UnsafePointer<Float>,
                                          weight: UnsafePointer<Float>,
                                          scale: Float,
                                          L: UnsafeMutablePointer<UInt8>) -> Int {
        let numNeighbours = Int(data.neighbours[offset])
        var bestD2 = Float.greatestFiniteMagnitude
        var gridIndex = -1
        for j in 1...numNeighbours {
            let candidate = Int(data.neighbours[offset + j])
            let pg = data.grid[candidate]
            var d2: Float = 0
            for i in 0..<8 {
                let q = Float(pg[i])
                let diff = scale * q - xval[i]
                d2 += weight[i] * diff * diff
            }
            if d2 < bestD2 {
                bestD2 = d2
                gridIndex = candidate
            }
        }
        let pg = data.grid[gridIndex]
        for i in 0..<8 { L[i] = UInt8((Int(pg[i]) - 1) / 2) }
        return gridIndex
    }

    static func writeIQ2XXSBlock(_ x: UnsafePointer<Float>,
                                 _ y: UnsafeMutablePointer<UInt8>,
                                 _ quantWeights: UnsafePointer<Float>) {
        let dOff = 0, qsOff = 2, groupSize = 32, kMaxQ = 3
        let data = iq2xxs

        var q2 = [UInt32](repeating: 0, count: 2 * (blockK / groupSize))
        var scales = [Float](repeating: 0, count: blockK / groupSize)
        var weight = [Float](repeating: 0, count: groupSize)
        var xval = [Float](repeating: 0, count: groupSize)
        var L = [UInt8](repeating: 0, count: groupSize)
        var Laux = [UInt8](repeating: 0, count: groupSize)
        var waux = [Float](repeating: 0, count: groupSize)
        var blockSigns = [UInt8](repeating: 0, count: 4)

        var hd = f16Bits(0)
        y[dOff] = UInt8(hd & 0xff); y[dOff + 1] = UInt8(hd >> 8)

        var sumx2: Float = 0
        for i in 0..<blockK { sumx2 += x[i] * x[i] }
        let sigma2 = sumx2 / Float(blockK)
        var maxScale: Float = 0

        for ib in 0..<(blockK / groupSize) {
            let xb = x + groupSize * ib
            let qw = quantWeights + groupSize * ib
            for i in 0..<groupSize {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).squareRoot()
                waux[i] = weight[i].squareRoot()
            }
            for k in 0..<4 {
                var nflip = 0
                var s: UInt8 = 0
                for i in 0..<8 {
                    let v = xb[8 * k + i]
                    if v >= 0 {
                        xval[8 * k + i] = v
                    } else {
                        xval[8 * k + i] = -v
                        nflip += 1
                        s |= UInt8(1 << i)
                    }
                }
                if nflip % 2 != 0 {
                    var imin = 0
                    var minW = weight[8 * k] * xb[8 * k] * xb[8 * k]
                    for i in 1..<8 {
                        let ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i]
                        if ax < minW {
                            minW = ax
                            imin = i
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin]
                    s ^= UInt8(1 << imin)
                }
                blockSigns[k] = s & 127
            }

            var maxV = xval[0]
            for i in 1..<groupSize { maxV = max(maxV, xval[i]) }
            if maxV < groupMaxEps {
                scales[ib] = 0
                for i in 0..<groupSize { L[i] = 0 }
                continue
            }

            var scale = xval.withUnsafeBufferPointer { xp in
                L.withUnsafeMutableBufferPointer { lp in
                    makeQP(n: groupSize, nmax: kMaxQ + 1, x: xp.baseAddress!,
                           L: lp.baseAddress!, quantWeights: weight)
                }
            }
            let effMax = scale * Float(kMaxQ)
            if effMax <= 0 {
                scales[ib] = 0
                for i in 0..<groupSize { L[i] = 0 }
                continue
            }

            var best: Float = 0
            for is_ in -6...6 {
                let id = (Float(2 * kMaxQ - 1) + Float(is_) * 0.1) / effMax
                let thisScale = 1 / id
                for k in 0..<4 {
                    var u = 0
                    for i in 0..<8 {
                        var l = nearestInt(0.5 * (id * xval[8 * k + i] - 1))
                        l = max(0, min(kMaxQ - 1, l))
                        Laux[8 * k + i] = UInt8(l)
                        u |= l << (2 * i)
                    }
                    if data.map[u] < 0 {
                        let offset = Int(-data.map[u]) - 1
                        xval.withUnsafeBufferPointer { xp in
                            waux.withUnsafeBufferPointer { wp in
                                Laux.withUnsafeMutableBufferPointer { lp in
                                    findBestNeighbour(
                                        offset: offset, data: data,
                                        xval: xp.baseAddress! + 8 * k,
                                        weight: wp.baseAddress! + 8 * k,
                                        scale: thisScale,
                                        L: lp.baseAddress! + 8 * k)
                                }
                            }
                        }
                    }
                }
                var sumqx: Float = 0
                var sumq2: Float = 0
                for i in 0..<groupSize {
                    let w = weight[i]
                    let q = Float(2 * Int(Laux[i]) + 1)
                    sumqx += w * xval[i] * q
                    sumq2 += w * q * q
                }
                if sumq2 > 0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2
                    best = scale * sumqx
                    for i in 0..<groupSize { L[i] = Laux[i] }
                }
            }

            if scale > 0 {
                let id = 1 / scale
                for k in 0..<4 {
                    var u = 0
                    for i in 0..<8 {
                        var l = nearestInt(0.5 * (id * xval[8 * k + i] - 1))
                        l = max(0, min(kMaxQ - 1, l))
                        u |= l << (2 * i)
                    }
                    var gridIndex = Int(data.map[u])
                    if gridIndex < 0 {
                        let offset = Int(-data.map[u]) - 1
                        gridIndex = xval.withUnsafeBufferPointer { xp in
                            waux.withUnsafeBufferPointer { wp in
                                L.withUnsafeMutableBufferPointer { lp in
                                    findBestNeighbour(
                                        offset: offset, data: data,
                                        xval: xp.baseAddress! + 8 * k,
                                        weight: wp.baseAddress! + 8 * k,
                                        scale: scale,
                                        L: lp.baseAddress! + 8 * k)
                                }
                            }
                        }
                    }
                    let pg = data.grid[gridIndex]
                    for i in 0..<8 { L[8 * k + i] = UInt8((Int(pg[i]) - 1) / 2) }
                }
                var sumqx: Float = 0
                var sumq2: Float = 0
                for i in 0..<groupSize {
                    let w = weight[i]
                    let q = Float(2 * Int(L[i]) + 1)
                    sumqx += w * xval[i] * q
                    sumq2 += w * q * q
                }
                if sumq2 > 0 { scale = sumqx / sumq2 }
            }

            if scale < 0 {
                scale = -scale
                for k in 0..<4 { blockSigns[k] = (~blockSigns[k]) & 127 }
            }

            for k in 0..<4 {
                var u = 0
                for i in 0..<8 { u |= Int(L[8 * k + i]) << (2 * i) }
                let gridIndex = Int(data.map[u])
                q2[2 * ib + 0] |= UInt32(gridIndex) << (8 * k)
                q2[2 * ib + 1] |= UInt32(blockSigns[k]) << (7 * k)
            }
            scales[ib] = scale
            maxScale = max(maxScale, scale)
        }

        if maxScale == 0 {
            for i in 0..<(blockK / 4) { y[qsOff + i] = 0 }
            return
        }

        let d = maxScale / 31
        hd = f16Bits(d)
        y[dOff] = UInt8(hd & 0xff); y[dOff + 1] = UInt8(hd >> 8)
        let id = 1 / d
        for ib in 0..<(blockK / groupSize) {
            var l = nearestInt(0.5 * (id * scales[ib] - 1))
            l = max(0, min(15, l))
            q2[2 * ib + 1] |= UInt32(l) << 28
        }
        for (i, word) in q2.enumerated() {
            let le = word.littleEndian
            y[qsOff + i * 4 + 0] = UInt8(le & 0xff)
            y[qsOff + i * 4 + 1] = UInt8((le >> 8) & 0xff)
            y[qsOff + i * 4 + 2] = UInt8((le >> 16) & 0xff)
            y[qsOff + i * 4 + 3] = UInt8((le >> 24) & 0xff)
        }
    }
}
