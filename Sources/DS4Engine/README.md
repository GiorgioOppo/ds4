# DS4Engine

`DS4Engine` è lo strato applicativo fra i dati portabili di `DS4Core`, il
backend GPU di `DS4Metal` e i client come la GUI. Il target non contiene viste:
coordina inferenza, strumenti, agenti, persistenza, modelli e nodi distribuiti.

## Dipendenze

```text
DS4Core ──┐
          ├── DS4Engine ──> GUI / servizi applicativi
DS4Metal ─┘
```

- `DS4Core`: GGUF, tokenizer, conversazione, sampling e formati condivisi.
- `DS4Metal`: runtime Metal, decoder, cache ed esecuzione dei layer.
- Foundation, Network, CryptoKit e Security sono usati solo nelle aree che ne
  hanno bisogno.

## Cartelle

- [`Inference`](Inference/README.md): API e actor che possiede il decoder.
- [`Distributed`](Distributed/README.md): protocollo, trasporto, coordinator e worker.
- [`Tools`](Tools/README.md): function calling, strumenti integrati e MCP.
- [`Persistence`](Persistence/README.md): checkpoint e cache persistenti.
- [`ModelManagement`](ModelManagement/README.md): download e sidecar del modello.
- [`Projects`](Projects/README.md): indice sicuro dei progetti importati.
- [`Agents`](Agents/README.md): profili e registro degli agenti.

## Regole architetturali

1. I tipi pubblici dell'inferenza vanno in `Inference/API`, non nella GUI.
2. Lo stato mutabile del decoder resta isolato da `InferenceService`.
3. I dati trasmessi in rete vivono in `Distributed/Protocol`; coordinator e
   worker ne consumano i tipi ma non ne definiscono il formato.
4. Un tool dichiara contratto ed esecuzione tramite `ToolRegistry`; le
   integrazioni riutilizzabili non vanno duplicate nei singoli built-in.
5. La persistenza non deve trattenere snapshot completi in RAM quando può
   elaborarli in streaming.
6. Le estensioni di un tipo principale seguono `Tipo+Responsabilita.swift`.

## Verifica delle modifiche

Dopo una modifica al target eseguire almeno `swift build --disable-sandbox` e i
test in `Tests/DS4CoreTests/Engine`. Le modifiche al protocollo richiedono anche
test di codifica/decodifica e un incremento di `Dist.protocolVersion` se non
sono compatibili con i nodi esistenti.
