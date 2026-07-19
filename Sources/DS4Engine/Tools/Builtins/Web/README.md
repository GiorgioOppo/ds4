**English** | [Italiano](README.it.md)

# Builtins/Web

Exposes limited web access to the model.

## Tools

- `web_search`: queries the configured endpoint (DuckDuckGo by default) and
  returns title, URL and snippet.
- `web_fetch`: downloads HTTP(S), converts HTML to text and supports windowing
  via `offset` for long pages.

## Flow and dependencies

Both delegate to [`WebClient`](../../Integrations/README.md), which applies
SSRF protection, redirect validation, timeouts and a body limit. The search
endpoint can be set with `DS4_SEARCH_URL` and must contain `%@`.

## Extension

Do not use `URLSession` directly in the built-in. Preserve the output limits
and do not follow URLs extracted from remote content without the same
validation. macOS ATS may reject insecure HTTP pages: HTTPS is the intended
path.
