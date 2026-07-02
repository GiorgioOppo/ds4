# Documentation

Detailed English documentation for DwarfStar and the pure-Swift DS4 engine.

- **`DOCUMENTAZIONE.md`** — user-facing guide: app workflow, settings, tools, agents, server, distributed inference, diagnostics, troubleshooting.
- **`ARCHITETTURA-MOTORE.md`** — engine internals: GGUF loading, tokenizer, decoder graph, MoE, NSA attention, streaming, expert cache, KV reuse, tools, distributed execution.
- **`CRITTOGRAFIA.md`** — encryption and export-compliance notes for App Store review: exempt TLS plus hashing only.
- **`UPSTREAM-SYNC.md`** — sync notes against upstream C (`antirez/ds4`): current baseline, recent commits, what is in scope, what has been ported, and how to repeat the comparison.
