import Foundation

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]   // names lowercased
    let body: Data
}

/// Serializes generations: the single-model engine can run only one at a time.

