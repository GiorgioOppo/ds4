import Foundation

func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

