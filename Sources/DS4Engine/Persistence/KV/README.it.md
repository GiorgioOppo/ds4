[English](README.md) | **Italiano**

# Persistence/KV

`DiskKVStore` conserva checkpoint del decoder indicizzati dal prefisso esatto
dei token, permettendo a chat, API stateless, sub-agent e worker di evitare un
prefill completo.

## File

- `DiskKVStore.swift`: configurazione, budget e stato.
- `+Index`: scansione entry, hit e strategia di eviction.
- `+Lookup`: ricerca del prefisso più lungo e restore.
- `+Store`: snapshot, scrittura atomica e manutenzione budget.
- `+Streaming`: import/export un layer alla volta.
- `+Serialization`: primitive del corpo binario.
- `GLM52DiskKVStore.swift`: la controparte GLM sui file `GKV1` nativi del
  motore — stesso lookup per prefisso, budget in token ed eviction con
  `KVCFile.evictionScore`, con hit/lastUsed in un `index.json` a lato così
  il formato wire del motore resta intatto.
- `LagunaDiskKVStore.swift`: la controparte Laguna sui file `LKV1` F16 nativi.
  La KV attiva resta in Metal; lo store ripristina sessioni fredde e scrive
  ogni plane sequenzialmente con `F_NOCACHE`, senza una copia cache-sized.

Il layout è documentato in [`FORMATO-CHECKPOINT.md`](FORMATO-CHECKPOINT.it.md).

## Flusso e dipendenze

Il lookup confronta modello e token prima del restore. Import e store rilasciano
ogni layer dopo l'uso, limitando il picco RAM; `F_NOCACHE` evita che i checkpoint
espellano i pesi caldi dalla page cache. Dipende da `DS4Core` per gli header e
da `DS4Metal` per `KVSnapshot`.

## Estensione

Preservare scrittura temporanea più rename, validazione completa prima
dell'import e budget sia in byte sia in token. Un formato incompatibile deve
essere versionato e i file vecchi devono fallire in modo sicuro.
