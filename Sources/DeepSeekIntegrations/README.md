# DeepSeekIntegrations

External-adapter target. Kept separate from the core build so the
failure modes of these systems (Slack outage, recorder-cassette
mismatch, sandbox-exec quirks) can't break the chat surface.

## Status matrix

| Adapter | What's here | What's missing |
|---|---|---|
| `HTTPRecorder` | Record/replay primitive that writes one JSON per request to a directory. | Hook into `OpenRouterAPI`'s `URLSession` (probably as a `URLProtocol` subclass). Test fixtures. |
| `Sandbox` | Generator for a starter `sandbox-exec` profile (deny default, read-only FS, no network). | Wire `ShellTool(useSandbox: true)` through the GUI toggle. Tune the profile for normal dev tasks. |
| `Slack/SlackNotifier` | One-shot incoming-webhook poster. | Full Slack bot: events API, OAuth, persistent sessions keyed by `(team_id, channel_id)`, server mode that lets a remote bot drive `InferenceService`. |
| `GitHub Actions` | Workflow templates under `.github/workflows/`. | Real ultrareview / autofix workflows that run the agent on PRs. |

See `TODO.md` for the items that are still open.
