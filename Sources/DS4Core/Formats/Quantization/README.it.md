[English](README.md) | **Italiano**

# Formats/Quantization

Primitive CPU portabili per conversione e requantizzazione dei pesi.

## File principali

- [`Half.swift`](Half.swift): conversioni f32/f16, inclusa la via software
  indipendente dall'architettura.
- [`Quantize.swift`](Quantize.swift): dequantizzazione Q8_0 e quantizzazione
  f32 -> Q4_K coerenti con i quantizzatori di riferimento ggml; dequant di
  riferimento Q2_K/Q5_K/Q6_K (esperti routed GLM 5.2) senza quantizzatore
  locale — i byte GGUF sono la fixture.
- [`QuantEncode.swift`](QuantEncode.swift) e
  [`QuantEncodeIQ2XXS.swift`](QuantEncodeIQ2XXS.swift): gli ENCODER per il
  writer GGUF (q8_0, q2_K, q4_K, q8_K, iq2_xxs; varianti reference e pesate
  imatrix), port Swift di `gguf-tools/quants.c` di ds4, fissato
  byte-per-byte contro il riferimento C compilato in `QuantEncodeTests`.
- [`GGUFRequantizer.swift`](GGUFRequantizer.swift): requantizzazione offline
  GGUF -> GGUF (la controparte in-process del `--tensor-type` selettivo di
  `gguf-tools/deepseek4-quantize.c` di ds4). Dequantizza i tensori sorgente
  (f32/f16/q8_0/q2_K/q4_K/q5_K/q6_K) in f32 e ri-codifica al tipo target via
  `QuantEncode`, scrivendo tramite `GGUFWriter`. Puro Swift, niente GPU; i
  tensori non gestibili passano invariati.

## Flusso

Il loader legge blocchi GGUF Q8_0, li converte in float e produce cache Q4_K
residenti per i percorsi configurati. I layout risultanti sono poi consumati dai
kernel di `DS4Metal`; questa cartella non effettua dispatch GPU.

## Regole di modifica

Layout, arrotondamento, scale e dimensioni di blocco sono parte del contratto con
i kernel. Ogni ottimizzazione deve mantenere test di parità numerica e casi per
valori limite; evitare API dipendenti da Metal o Accelerate.
