import Foundation
import CoreGraphics
import ImageIO
import DS4Core

struct DeepSeekV4ImagePatches {
    let gridHeight: Int
    let gridWidth: Int
    let values: [Float]
}

/// ImageIO handles the input format and EXIF orientation; resizing and patch
/// ordering follow antirez/ds4's ds4_image_preprocess_deepseek4 reference.
enum DeepSeekV4ImagePreprocessor {
    static func preprocess(data: Data) throws -> DeepSeekV4ImagePatches {
        guard !data.isEmpty, data.count <= 40 * 1024 * 1024,
              let source = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = props[kCGImagePropertyPixelWidth] as? Int,
              let height = props[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0, width <= 16_384, height <= 16_384,
              width * height <= 40_000_000 else {
            throw DeepSeekV4VisionError.invalidImage("formato non leggibile o dimensioni eccessive (40 MB, 40 megapixel)")
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(width, height),
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw DeepSeekV4VisionError.invalidImage("decodifica non riuscita")
        }
        var rgba = [UInt8](repeating: 255, count: image.width * image.height * 4)
        let drew = rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard let context = CGContext(data: bytes.baseAddress, width: image.width,
                height: image.height, bitsPerComponent: 8, bytesPerRow: image.width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: image.width, height: image.height))
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            return true
        }
        guard drew else { throw DeepSeekV4VisionError.invalidImage("buffer RGB non disponibile") }
        return try preprocess(rgba: rgba, width: image.width, height: image.height)
    }

    static func targetSize(width: Int, height: Int) throws -> (width: Int, height: Int) {
        guard width > 0, height > 0, width <= 16_384, height <= 16_384 else {
            throw DeepSeekV4VisionError.invalidImage("dimensioni non valide")
        }
        var plannedWidth = min(width, height * 8), plannedHeight = height
        let pixels = plannedWidth * plannedHeight
        if pixels < 147_456 {
            let scale = sqrt(147_456.0 / Double(pixels))
            plannedWidth = max(1, Int(Double(plannedWidth) * scale))
            plannedHeight = max(1, Int(Double(plannedHeight) * scale))
        }
        var bestWidth = (plannedWidth + 13) / 14 * 14
        var bestHeight = (plannedHeight + 13) / 14 * 14
        func tokenCount() -> Int {
            let h = (bestHeight / 14 + 2) / 3, w = (bestWidth / 14 + 2) / 3
            let evenH = h + (h & 1)
            return evenH * (w + 1) + 2 + (((evenH / 2 * (w + 1)) & 1) * 2)
        }
        var budget = 381
        while tokenCount() > 381 {
            guard budget > 4 else { throw DeepSeekV4VisionError.invalidImage("impossibile rispettare il limite di 384 token") }
            let ratio = Double(plannedHeight) / Double(plannedWidth)
            let maxWFloat = sqrt((Double(budget) - 2) / ratio + 0.25) - 0.5
            let maxHFloat = maxWFloat * ratio
            if maxWFloat < 1 {
                bestWidth = 42
                bestHeight = ((budget - 2) / 2 & ~1) * 42
            } else if maxHFloat < 2 {
                bestWidth = ((budget - 2) / 2 - 1) * 42
                bestHeight = 84
            } else {
                let maxW = Int(floor(maxWFloat)), maxH = Int(floor(maxHFloat)) & ~1
                let scale = min(Double(maxW * 42) / Double(plannedWidth), Double(maxH * 42) / Double(plannedHeight))
                bestWidth = Int(floor(Double(plannedWidth) * scale / 14)) * 14
                bestHeight = Int(floor(Double(plannedHeight) * scale / 14)) * 14
            }
            guard bestWidth > 0, bestHeight > 0 else {
                throw DeepSeekV4VisionError.invalidImage("proporzioni non supportate")
            }
            budget -= 1
        }
        return (bestWidth, bestHeight)
    }

    static func preprocess(rgba: [UInt8], width: Int, height: Int) throws -> DeepSeekV4ImagePatches {
        let target = try targetSize(width: width, height: height)
        guard rgba.count == width * height * 4 else {
            throw DeepSeekV4VisionError.invalidImage("buffer RGB incompleto")
        }
        var contentWidth = target.width, contentHeight = target.height
        var offsetX = 0, offsetY = 0
        if width < height * 8 {
            let scale = min(Double(target.width) / Double(width), Double(target.height) / Double(height))
            contentWidth = max(1, min(target.width, Int((Double(width) * scale).rounded(.toNearestOrEven))))
            contentHeight = max(1, min(target.height, Int((Double(height) * scale).rounded(.toNearestOrEven))))
            offsetX = Int((Double(target.width - contentWidth) * 0.5).rounded(.toNearestOrEven))
            offsetY = Int((Double(target.height - contentHeight) * 0.5).rounded(.toNearestOrEven))
        }
        // Separable bicubic filtering has the same sampling centers and
        // antialias support as upstream. Keep Double intermediates and round
        // only the final pixel, as its direct two-dimensional convolution does.
        let xWeights = weights(source: width, destination: contentWidth)
        let yWeights = weights(source: height, destination: contentHeight)
        var horizontal = [Double](repeating: 0, count: height * contentWidth * 3)
        for y in 0..<height {
            if y & 63 == 0 { try Task.checkCancellation() }
            for x in 0..<contentWidth {
                let sample = xWeights[x]
                for (sourceX, weight) in sample {
                    let src = (y * width + sourceX) * 4, dst = (y * contentWidth + x) * 3
                    for channel in 0..<3 { horizontal[dst + channel] += Double(rgba[src + channel]) * weight }
                }
            }
        }
        var canvas = [Float](repeating: Float(127) / 127.5 - 1, count: target.width * target.height * 3)
        for y in 0..<contentHeight {
            if y & 31 == 0 { try Task.checkCancellation() }
            for x in 0..<contentWidth {
                var pixel = (0.0, 0.0, 0.0)
                for (sourceY, weight) in yWeights[y] {
                    let src = (sourceY * contentWidth + x) * 3
                    pixel.0 += horizontal[src] * weight
                    pixel.1 += horizontal[src + 1] * weight
                    pixel.2 += horizontal[src + 2] * weight
                }
                let dst = ((y + offsetY) * target.width + x + offsetX) * 3
                canvas[dst] = Float(min(255, max(0, pixel.0.rounded()))) / 127.5 - 1
                canvas[dst + 1] = Float(min(255, max(0, pixel.1.rounded()))) / 127.5 - 1
                canvas[dst + 2] = Float(min(255, max(0, pixel.2.rounded()))) / 127.5 - 1
            }
        }
        let gridHeight = target.height / 14, gridWidth = target.width / 14
        var patches = [Float]()
        patches.reserveCapacity(gridHeight * gridWidth * 588)
        for py in 0..<gridHeight {
            for px in 0..<gridWidth {
                for channel in 0..<3 {
                    for y in 0..<14 {
                        for x in 0..<14 {
                            patches.append(canvas[((py * 14 + y) * target.width + px * 14 + x) * 3 + channel])
                        }
                    }
                }
            }
        }
        return .init(gridHeight: gridHeight, gridWidth: gridWidth, values: patches)
    }

    private static func weights(source: Int, destination: Int) -> [[(Int, Double)]] {
        let scale = Double(source) / Double(destination)
        let filter = scale >= 1 ? 1 / scale : 1, support = scale >= 1 ? 2 * scale : 2
        func cubic(_ raw: Double) -> Double {
            let x = abs(raw)
            if x < 1 { return (1.5 * x - 2.5) * x * x + 1 }
            if x < 2 { return ((-0.5 * x + 2.5) * x - 4) * x + 2 }
            return 0
        }
        return (0..<destination).map { d in
            let center = scale * (Double(d) + 0.5)
            let lo = max(0, Int(center - support + 0.5)), hi = min(source, Int(center + support + 0.5))
            let samples = (lo..<hi).map { ($0, cubic((Double($0) + 0.5 - center) * filter)) }
            let sum = samples.reduce(0) { $0 + $1.1 }
            return samples.map { ($0.0, $0.1 / sum) }
        }
    }
}
