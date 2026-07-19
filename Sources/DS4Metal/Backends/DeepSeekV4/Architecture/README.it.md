[English](README.md) | **Italiano**

# DeepSeekV4/Architecture

Geometria e dimensioni derivate dei profili DeepSeek-V4 serviti dai kernel
Metal. Il profilo Flash statico resta disponibile come compatibilità sorgente,
ma la costruzione model-aware usa la geometria instance-based.

## File principali

- [`DSV4Shape.swift`](DSV4Shape.swift): layer, head, expert, rapporti di
  compressione e costanti Flash legacy.
- [`DSV4RuntimeGeometry.swift`](DSV4RuntimeGeometry.swift): geometria runtime
  instance-based derivata dal profilo Flash/Pro o dalla configurazione GGUF.
- [`DSV4Dims.swift`](DSV4Dims.swift): dimensioni runtime derivate e flag delle
  fusioni/kernel configurabili.
- [`RopeParams.swift`](RopeParams.swift): parametri per RoPE e scaling del contesto.

## Flusso e dipendenze

Il loader valida i metadati GGUF in `DeepSeekV4Configuration`; da questa viene
costruita una `DSV4RuntimeGeometry`. Decoder, scratch, grafo e wrapper kernel
possono così condividere dimensioni, compressione e RoPE del profilo selezionato.
`DSV4Shape` conserva l'API Flash precedente per test e chiamanti legacy; non
deve essere usato per dimensionare un GGUF Pro. Il profilo Pro Q2 locale usa
61 layer, 384 esperti e i propri rapporti di compressione attraverso la stessa
geometria runtime.

## Regole di modifica

Non correggere una forma incompatibile con fallback silenziosi. Distinguere
costanti del modello da tuning hardware. Ogni nuovo campo deve indicare unità,
origine e consumatori; aggiornare validazione e test di allocazione insieme.
