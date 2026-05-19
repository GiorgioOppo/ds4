import Foundation

/// Where a model lives. The chat layer talks to whichever endpoint
/// is currently `loaded` on `ModelState` without caring about the
/// flavour — the inference plumbing on the other side maps the
/// endpoint to a concrete backend (today: local Metal transformer;
/// future: HTTP/SSE remote inference).
///
/// Codable so the `ModelLibrary` can persist a recents list under
/// `Application Support/.../models.json`, and Hashable so SwiftUI
/// pickers / dictionaries can use it directly.
enum ModelEndpoint: Codable, Hashable {
    /// On-disk converted V4 directory: contains tokenizer.json,
    /// config.json, and `model-*.safetensors` shards. The whole
    /// load path (`InferenceService.loadModel(at:)`) operates on
    /// the wrapped URL.
    case localDirectory(path: String)

    /// Remote OpenAI-compatible model hosted through OpenRouter.
    /// `modelID` is the slug from `GET /api/v1/models` (e.g.
    /// "anthropic/claude-3.5-sonnet", "deepseek/deepseek-r1").
    /// The API key lives in the Keychain under
    /// `KeychainAccount.openRouterAPIKey`; not embedded here so
    /// rotating the key doesn't invalidate every persisted
    /// `ConfiguredModelEntry`.
    case openRouter(modelID: String)

    /// Anthropic native endpoint (`api.anthropic.com/v1/messages`).
    /// `modelID` is the Anthropic model slug (e.g.
    /// "claude-3-5-sonnet-20241022"). Picked over the OpenRouter
    /// route when the user wants prompt caching (TODO §10.4) — the
    /// OpenRouter body doesn't expose `cache_control`. API key lives
    /// in the Keychain under `KeychainAccount.anthropicAPIKey`.
    case anthropic(modelID: String)

    /// Stable identifier for SwiftUI ForEach / set membership.
    /// Built from the case payload so two values targeting the
    /// same model compare equal.
    var id: String {
        switch self {
        case .localDirectory(let path):   return "local::\(path)"
        case .openRouter(let modelID):    return "openrouter::\(modelID)"
        case .anthropic(let modelID):     return "anthropic::\(modelID)"
        }
    }

    /// User-facing label. For local directories it's the trailing
    /// folder name; for OpenRouter it's the slug minus the
    /// provider prefix ("claude-3.5-sonnet" rather than the full
    /// "anthropic/claude-3.5-sonnet" so the picker stays readable);
    /// for Anthropic native we display the bare model id.
    var displayName: String {
        switch self {
        case .localDirectory(let path):
            let url = URL(fileURLWithPath: path)
            let name = url.lastPathComponent
            return name.isEmpty ? path : name
        case .openRouter(let modelID):
            if let slash = modelID.firstIndex(of: "/") {
                return String(modelID[modelID.index(after: slash)...])
            }
            return modelID
        case .anthropic(let modelID):
            return modelID
        }
    }

    /// Long-form description used in tooltips and the picker
    /// secondary label. For local directories shows the full path;
    /// for OpenRouter shows the full slug so the provider stays
    /// visible; for Anthropic the prefix marks the native route
    /// distinct from the OpenRouter relay of the same model.
    var subtitle: String {
        switch self {
        case .localDirectory(let path):   return path
        case .openRouter(let modelID):    return "openrouter · \(modelID)"
        case .anthropic(let modelID):     return "anthropic · \(modelID)"
        }
    }

    /// SF Symbol the toolbar picker uses for this endpoint kind.
    /// Distinct symbols make the local-vs-remote distinction
    /// obvious at a glance.
    var iconName: String {
        switch self {
        case .localDirectory: return "internaldrive"
        case .openRouter:     return "cloud"
        case .anthropic:      return "cloud.fill"
        }
    }

    /// True when the endpoint represents a remote backend — used
    /// by the chat path to bypass the token-level fast-delta cache
    /// (it's pointless when the actual KV cache lives on the
    /// provider's GPU, not in this process).
    var isRemote: Bool {
        switch self {
        case .localDirectory: return false
        case .openRouter:     return true
        case .anthropic:      return true
        }
    }
}
