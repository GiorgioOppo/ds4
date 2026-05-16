import Foundation

/// Slack integration — scaffolded only.
///
/// opencode's `packages/slack` exposes the agent as a Slack bot that
/// converses in a channel and runs tools on a remote host. That
/// design needs:
///   1. A long-running server process that accepts Slack Events API
///      callbacks (`https://api.slack.com/events`).
///   2. Persistent session state keyed by Slack `team_id / channel_id`.
///   3. A bridge from `InferenceService` (today a `@MainActor` GUI
///      thing) to a headless server.
///
/// None of that exists yet in this project; today we only ship a
/// one-shot incoming-webhook notifier so the GUI / CLI can post
/// "long generation completed" / "error" pings into a channel. The
/// full bot is tracked in TODO.md.
public struct SlackNotifier: Sendable {
    public let webhookURL: URL

    public init(webhookURL: URL) {
        self.webhookURL = webhookURL
    }

    public init?(webhookURLString: String) {
        guard let url = URL(string: webhookURLString),
              url.scheme?.hasPrefix("http") == true else { return nil }
        self.webhookURL = url
    }

    public func post(text: String) async throws {
        var req = URLRequest(url: webhookURL)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = ["text": text]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (_, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse,
              200..<300 ~= http.statusCode else {
            throw NSError(domain: "SlackNotifier",
                          code: (response as? HTTPURLResponse)?.statusCode ?? 0,
                          userInfo: [NSLocalizedDescriptionKey: "Slack webhook failed"])
        }
    }
}
