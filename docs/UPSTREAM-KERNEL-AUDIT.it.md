# Kernel e modelli del branch ternary-bonsai-2-support

Confronto del 21 settembre 2026. Riferimento C fissato a
[`acf5c16bb6a01f8c4f027a0db265ae270d8907ab`](https://github.com/GiorgioOppo/ds4/tree/acf5c16bb6a01f8c4f027a0db265ae270d8907ab),
Swift a `b4a9ee67` più le due ottimizzazioni DeepSeek in lavorazione.
È un'ispezione del codice: i benchmark upstream citati nei documenti C non sono
misure dell'app Swift. Il branch remoto può cambiare dopo questa revisione.

## Modelli e stato del port

| Modello | Implementazione nel branch C | Stato del runtime Swift |
|---|---|---|
| DeepSeek V4 Flash/Pro e Flash Vision Experimental | Backend DeepSeek e shader `dense`, `moe`, `dsv4_*`, `deepseek4_vision` | Backend presente; supporto limitato ai profili e formati accettati dal loader Swift. MXFP4 e Pro Q4 split non diventano eseguibili copiando i kernel. |
| DeepSeek V4.1 Flash | Grafo e tokenizer distinti; `dsv41.metal`, Engram e pesi BF16. Metal testo/immagini; CUDA testo | Backend V4.1 assente. Il riconoscimento del prefisso DeepSeek non implica compatibilità con V4. |
| GLM 5.2 | Backend GLM-DSA, Metal/CUDA/ROCm | Eseguibile con gate attivo, ma il gate nel codice dichiara ancora sperimentale la parità completa dei logits. |
| GLM 5.3 Flash | Backend `glm5-next` con KDA, `glm53_bf16`, `glm53_kda` e vision dedicata | Backend `glm5-next` assente. |
| GLM 5.3 full | Supportato sul percorso GLM-DSA condiviso | Compatibilità non verificata: il backend Swift è validato e nominato GLM5.2; la condivisione dell'identificatore non certifica il nuovo checkpoint. |
| Qwen3.8 Flash Next | Architettura interna `qwen4exp`; HC, GDN, n-gram, attenzione sparsa e MoE in `qwen4.metal`; encoder Qwen3-VL separato | Famiglia riconosciuta ma esecuzione rifiutata: mancano backend e contratti del checkpoint. Il nome del file C non significa supporto generico a Qwen4. |
| Ternary Bonsai 2 27B | Backend dedicato CPU/Metal; `metal/bonsai.metal`; PQ2_0 e PTQ1_0 nativi; immagini su Metal | Mancano codec GGUF, loader, decoder e sessione specifici. `qwen35` ricade nel Qwen non eseguibile. |

Il dispatch C è in
[`ds4.c`](https://github.com/GiorgioOppo/ds4/blob/acf5c16bb6a01f8c4f027a0db265ae270d8907ab/ds4.c#L7274);
i limiti per modello in
[`docs/MODELS.md`](https://github.com/GiorgioOppo/ds4/blob/acf5c16bb6a01f8c4f027a0db265ae270d8907ab/docs/MODELS.md).
Sul lato Swift fanno fede
[`BackendSelector.swift`](../Sources/DS4Engine/Runtime/Common/BackendSelector.swift),
[`GGUFTypes.swift`](../Sources/DS4Core/Formats/GGUF/GGUFTypes.swift) e i gate di runtime,
non la sola presenza di una voce nel catalogo.
In particolare [`GLM52RuntimeGate.swift`](../Sources/DS4Metal/Backends/GLM52/Engine/GLM52RuntimeGate.swift)
distingue l'abilitazione anticipata dalla validazione completa.

Laguna e Kimi K3 non compaiono come backend nel dispatch di questo branch C.
Nel repository Swift Laguna ha un percorso sperimentale opt-in e Kimi è
disabilitato. La primitiva “Kimi Delta Attention” usata da GLM non certifica un
decoder Kimi K3; le fixture “Qwen4 MINI” non sono modelli pubblicati supportati.

## Perché Bonsai richiede un backend proprio

Il loader C accetta il checkpoint denso `qwen35` con 64 layer, embedding 5120,
FFN 17408 e gruppi di tre layer Gated DeltaNet più uno di full attention.
Questo contratto non equivale al supporto generico di tutti i modelli Qwen3.5.
Non ci sono MoE, Hyper-Connection o streaming degli esperti.

- **PQ2_0, tipo GGUF 142:** 128 pesi in 34 byte, scala FP16 e codici a 2 bit
  per i valori `{-1, 0, 1, 2}`.
- **PTQ1_0, tipo 143:** 128 pesi in 28 byte. Ha un impacchettamento ternario
  distinto da TQ1/TQ2 e Q2_K; la decodifica deve preservare il wrap a 8 bit
  prima dell'estrazione dei trit.
- Le matrici folded richiedono trasformazioni Hadamard normalizzate di
  dimensione 1024, con segni e permutazioni del checkpoint verificati.
- Lo stato ricorrente GDN cambia il significato di reset, rewind e riuso del
  prefisso. Non basta riutilizzare la cache KV DeepSeek.

Il riferimento completo è
[`bonsai_bind.inc`](https://github.com/GiorgioOppo/ds4/blob/acf5c16bb6a01f8c4f027a0db265ae270d8907ab/bonsai_bind.inc)
insieme a
[`docs/BONSAI.md`](https://github.com/GiorgioOppo/ds4/blob/acf5c16bb6a01f8c4f027a0db265ae270d8907ab/docs/BONSAI.md).
La dimensione documentata dei pesi PQ2 è circa 6,71 GiB; si aggiungono KV,
stato GDN, scratch ed eventuale encoder immagini. È quindi un candidato
pratico da validare sul Mac da 32 GB, senza dedurne una velocità prima delle misure.

## Ottimizzazioni da conservare nel port Bonsai

| Percorso | Ottimizzazione C | Vincoli importanti |
|---|---|---|
| Proiezioni decode | Lettura diretta PQ2/PTQ e riuso di attivazioni tra righe | Codec esatto e trattamento delle righe parziali; nessuna dequantizzazione permanente dell'intero modello. |
| FFN | Gate/up e SiLU nello stesso kernel | Tipi, forme e tabelle dei segni Hadamard devono coincidere. |
| Prefill PQ2 | Tile 64 output × 32 token × 32 K, da 16 token per matrici grandi | Accumulo FP32; associazione diversa dal GEMV, da verificare con errori numerici e logits. |
| Prefill PTQ | Blocchi da 4/8 token, da 4 token per matrici grandi | Percorso specifico PTQ; non riconversione in PQ2. |
| GDN | Stato 128 mantenuto nei registri durante la scansione | Token causali in ordine e stato completo verificato dopo ogni chunk. |
| Norme e Hadamard | Meno barriere e lavoro a blocchi | Conservare ordine delle riduzioni e normalizzazione. |
| Full attention prefill | Fino a otto query per gruppo, scratch limitato | Ogni query conserva il proprio limite causale. |
| BF16 alpha/beta | Proiezioni accoppiate con accumulatori indipendenti | Solo proiezioni non ruotate e forme compatibili. |

Le selezioni effettive sono in
[`ds4_bonsai_metal.m`](https://github.com/GiorgioOppo/ds4/blob/acf5c16bb6a01f8c4f027a0db265ae270d8907ab/ds4_bonsai_metal.m),
la matematica in
[`metal/bonsai.metal`](https://github.com/GiorgioOppo/ds4/blob/acf5c16bb6a01f8c4f027a0db265ae270d8907ab/metal/bonsai.metal).
La riduzione dei dispatch e dei trasferimenti intermedi spiega perché un
runtime possa essere più veloce anche quando entrambi usano Metal: il confronto
non isola la velocità del linguaggio C rispetto a Swift.

## Sequenza di implementazione proposta

1. Codec PQ2/PTQ e binding del solo checkpoint Bonsai supportato, con oracle
   indipendenti per byte, scale e Hadamard.
2. Tokenizer/template, grafo testo, stato GDN, reset/replay e parità prefill/decode
   contro il runtime C e i riferimenti Prism già presenti nel branch.
3. Specializzazioni Metal e benchmark sullo stesso GGUF, prompt, contesto e
   numero di token. Distinguere tempi dei kernel da token/secondo dell'intero modello.
4. Integrazione nel selettore, catalogo e impostazioni Swift in base alle capacità
   reali; rendere selezionabile il modello dopo la validazione.
5. Vision Bonsai: encoder Qwen3-VL BF16 o Q8/F16, preprocessing, embedding
   sostitutivi e MRoPE tridimensionale, più test del riuso della sessione.

La vision Bonsai non usa l'encoder DeepSeek Vision Experimental già portato:
ha geometria e posizioni diverse. CUDA/ROCm, MTP, distribuzione, streaming SSD
e cache serializzate non sono implementati per Bonsai nel riferimento C.

DeepSeek V4.1, GLM5.3 e Qwen3.8 richiedono verifiche e integrazioni distinte,
con contratti specifici di loader, tokenizzazione e grafo e riuso dei componenti
compatibili. Le fusioni Swift DeepSeek output-B/HC e Q-A/KV
attualmente in verifica richiedono rispettivamente HC=4 e la forma
4096→1024+512: non abilitano Bonsai, che usa embedding 5120 e PQ2/PTQ.

## Relazione con il precedente branch attention

`aprojq4-dense-attention` a `b8507c9f` e questo branch a `acf5c16b` divergono
da `6289c516`: il secondo non include automaticamente tutte le ottimizzazioni
del primo. Il commit `c9474f66` del port Q4 appartiene al ramo precedente.
In `acf5c16b` mancano sia output-B/HC Q4 sia il paired prefill Q-A/KV con RHS
F16. Il percorso Q8 output-B/HC e i due helper aritmetici Q4 confrontati sono
invece identici. Le due ottimizzazioni Swift conservano quindi come riferimento
`b8507c9f`, mentre questo audit usa `acf5c16b` per i nuovi modelli.
