# Documentazione di DwarfStar

Indice della documentazione tecnica e operativa. Ogni cartella del codice e dei
test ha inoltre un `README.md` locale con responsabilità, dipendenze, file
principali e regole di modifica.

## Per iniziare

- [README del progetto](../README.md) — panoramica, avvio rapido e riferimento
  completo della configurazione.
- [DOCUMENTAZIONE.md](DOCUMENTAZIONE.md) — guida ampia all'app, workflow,
  pannelli e troubleshooting.
- [STRUTTURA-PROGETTO.md](STRUTTURA-PROGETTO.md) — moduli, dipendenze e mappa
  delle cartelle.
- [ARCHITETTURE-SUPPORTATE.md](ARCHITETTURE-SUPPORTATE.md) — matrice dei
  backend, rilevamento GGUF, capacità e regole per introdurre Qwen.
- [GUIDA-SVILUPPO.md](GUIDA-SVILUPPO.md) — workflow per modificare codice,
  documentazione, kernel e progetto Xcode.

## Motore e inferenza

- [`architectures/`](architectures/README.md) — documentazione separata per
  DeepSeek V4 e per il backend Qwen ancora in preparazione.
- [PIPELINE-INFERENZA.md](PIPELINE-INFERENZA.md) — ciclo completo da prompt a
  token, ownership dello stato, prefill, decode e tool loop.
- [ARCHITETTURA-MOTORE.md](ARCHITETTURA-MOTORE.md) — dettagli del modello,
  GGUF, tokenizer, NSA, MoE, quantizzazione e grafo.
- [BACKEND-METAL.md](BACKEND-METAL.md) — runtime, tensori, command buffer,
  wrapper, kernel generati e validazione numerica.
- [DS4CORE-INFERENCE.md](DS4CORE-INFERENCE.md) — riferimento compatto ai
  componenti inference-facing privi di dipendenza GPU.
- [`DS4Engine/Inference/FLUSSO-INFERENZA.md`](../Sources/DS4Engine/Inference/FLUSSO-INFERENZA.md)
  — dettaglio del servizio applicativo e delle sue estensioni.

## Configurazione e prestazioni

- [Configuration Reference](../README.md#configuration-reference) — tabella
  autorevole di impostazioni GUI, `DS4_*`, server, distribuzione e MCP.
- [CONFIGURAZIONE-E-PROFILI.md](CONFIGURAZIONE-E-PROFILI.md) — precedenza,
  momento di lettura, profili qualità/prestazioni e regole per nuovi knob.
- [VALUTAZIONE-DEMO-PERF.md](VALUTAZIONE-DEMO-PERF.md) — misure storiche della
  demo, colli di bottiglia e runbook A/B.
- [SELF-SPECULATIVE.md](SELF-SPECULATIVE.md) — design e misure datate del
  decode self-speculative, attualmente opt-in sperimentale.

I file con misure riportano data e macchina: sono fotografie sperimentali, non
default universali.

## Distribuzione

- [INFERENZA-DISTRIBUITA.md](INFERENZA-DISTRIBUITA.md) — topologia
  orizzontale e verticale, setup, file, KV, sicurezza e protocollo v11.
- [EXPERT_PARALLELISM.md](EXPERT_PARALLELISM.md) — stato implementativo,
  costi e validazione della scissione verticale.
- [`Distributed/PROTOCOLLO.md`](../Sources/DS4Engine/Distributed/PROTOCOLLO.md)
  — messaggi e invarianti vicino al codice.

## GUI, server, strumenti e dati

- [GUI-SERVER-E-API.md](GUI-SERVER-E-API.md) — feature SwiftUI, motore
  condiviso, HTTP/SSE e regole per nuovi endpoint.
- [STRUMENTI-AGENTI-MCP.md](STRUMENTI-AGENTI-MCP.md) — registro tool, agenti,
  sub-agent, trasporti MCP e sicurezza.
- [`Chat/FLOW.md`](../Sources/DwarfStar/Features/Chat/FLOW.md) — flusso della
  chat vicino alla feature.
- [`Server/HTTP-API.md`](../Sources/DwarfStar/Features/Server/HTTP-API.md) —
  contratti HTTP locali e mapping degli endpoint.
- [`GESTIONE-MODELLI.md`](../Sources/DS4Engine/ModelManagement/GESTIONE-MODELLI.md)
  — download, token, sidecar e lifecycle dei modelli.
- [`FORMATO-CHECKPOINT.md`](../Sources/DS4Engine/Persistence/KV/FORMATO-CHECKPOINT.md)
  — checkpoint KV e compatibilità persistente.
- [`SICUREZZA-PERCORSI.md`](../Sources/DS4Engine/Projects/SICUREZZA-PERCORSI.md)
  e [`Tools/SICUREZZA.md`](../Sources/DS4Engine/Tools/SICUREZZA.md) — confini
  filesystem e sicurezza dei tool.

## Test, rilascio e conformità

- [TESTING-E-VALIDAZIONE.md](TESTING-E-VALIDAZIONE.md) — strategia, comandi,
  skip GPU, parità e checklist.
- [`Tests/METAL-TESTS.md`](../Tests/METAL-TESTS.md) — requisiti e convenzioni
  specifiche per i test Metal.
- [CRITTOGRAFIA.md](CRITTOGRAFIA.md) — inventario crittografico, trasporti in
  chiaro e note di export compliance con fonti ufficiali.
- [UPSTREAM-SYNC.md](UPSTREAM-SYNC.md) — fotografia datata del confronto con
  il progetto C upstream e procedura per aggiornarla.
- [`packaging/README.md`](../packaging/README.md) — bundle, firma ed
  entitlements.

## Sorgenti autorevoli

- I kernel si modificano in [`metal/`](../metal/README.md), non nel file Swift
  generato.
- Il template tool-calling di riferimento è in
  [`templates/`](../templates/README.md).
- Gli script di build e analisi sono descritti in
  [`scripts/`](../scripts/README.md).
- La mappa completa dei sorgenti parte da [`Sources/README.md`](../Sources/README.md).
- La mappa dei test parte da [`Tests/README.md`](../Tests/README.md).

## Regole di manutenzione

Quando cambia un comportamento:

1. aggiornare il README della cartella proprietaria;
2. aggiornare il documento tematico corrispondente;
3. mantenere esempi e default coerenti con il codice;
4. dichiarare data e hardware per misure prestazionali;
5. distinguere funzioni operative, sperimentali e soltanto progettate;
6. verificare collegamenti relativi e percorsi dopo ogni spostamento.

I Markdown collocati dentro i target sono esclusi automaticamente da SwiftPM e
da XcodeGen: possono restare accanto al codice senza entrare nel binario.
