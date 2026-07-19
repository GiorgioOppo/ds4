[English](README.md) | **Italiano**

# Tests/DS4CoreTests

Test di parità e unitari per `DS4Core`, `DS4Metal` e `DS4Engine` con XCTest.
Ogni file prende di mira un kernel o un componente e ne confronta l'output con
un riferimento, spesso un'implementazione CPU fedele.

Le aree coperte includono i kernel Metal (`Graph*`, `Metal*`, `MoE`),
`StreamingDecoder`, il loader GGUF, `Half`, `KVCFile`, i tokenizer, i sampler,
i protocolli chat/tool di DeepSeek e GLM, lo schema GLM e il layout DSA,
`SSDCachePlan`, il KV store su disco, il protocollo distribuito, MCP, la
cache dei progetti, il registro dei tool e le utility del downloader come
`sha256Hex` e la mappa dei target.

Quando si tocca un invariante numerico, come il **ring raw-KV**, aggiungi un
test che genera più di `nSWA` token con e senza la feature e ne asserisce
l'uguaglianza.

## Mappa delle directory

- [`Core/`](Core/README.it.md): test unitari deterministici solo CPU.
- [`Metal/`](Metal/README.it.md): kernel, grafi, decoder, loader e runtime.
- [`Engine/`](Engine/README.it.md): servizi, persistenza, protocolli, progetti e
  tool.

I test seguono l'ownership di produzione: aggiungi un nuovo caso accanto
all'area del componente, non alla radice di questa directory. Usa file
temporanei e dipendenze iniettate; non fare mai affidamento su credenziali,
repository, modello scaricato o accesso di rete dell'utente. Le convenzioni di
skip su GPU sono documentate in [`../METAL-TESTS.md`](../METAL-TESTS.it.md).
