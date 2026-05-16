# GGUF support

Stato attuale: **lettore MVP — solo header parsing + pass-through
dtypes**. Ottenere "inferenza GGUF end-to-end" è un percorso a più
fasi; questo doc spiega cosa è dentro, cosa è fuori, e quali sono i
prossimi step.

## Cosa funziona oggi

`Sources/DeepSeekKit/GGUF.swift` (~300 LOC) implementa il parser
metadata GGUF v2/v3 per intero:

- Magic `"GGUF"` + version check (accetta v2 e v3, le sole in
  produzione).
- Key/Value metadata di tutti i tipi previsti (`u8/i8/u16/i16/u32/
  i32/u64/i64/f32/f64/bool/string/array`).
- Tabella tensor info con `(name, n_dims, shape[], ggml_type,
  offset)`.
- Padding all'allineamento globale (`general.alignment`, default 32).
- Calcolo dell'offset assoluto e dei byte/tensor.

`Sources/DeepSeekKit/GGUFLoader.swift` (~100 LOC) wraps il file con
mmap + MTLBuffer (stesso pattern di `SafeTensorsFile`):

- `GGUFFile.init(url:)` → mmap + parse header.
- `tensorNames`, `info(name:)` → ispezione strutturale.
- `load(name:) -> Tensor` → ritorna una vista zero-copy nel buffer
  mmap per i dtype pass-through:
  - `GGML_TYPE_F32`
  - `GGML_TYPE_F16`
  - `GGML_TYPE_BF16`
  - `GGML_TYPE_I32`
  - `GGML_TYPE_I8`

## Cosa NON funziona oggi

### 1. Lettura di tensor quantizzati

GGUF è quasi esclusivamente quantizzato in pratica. I dtype principali
(`Q4_0`, `Q4_K`, `Q4_K_M`, `Q8_0`, `Q5_K`, `Q6_K`) richiedono kernel
Metal di dequantizzazione che NON sono ancora implementati.

Quando un caller chiama `load(name:)` su un tensor quantizzato, viene
sollevato `GGUFError.unsupportedType`. Il caller può comunque chiamare
`info(name:)` per ottenere `(shape, type, absoluteOffset, byteCount)`
e processare i byte manualmente.

### 2. Lettura di tensor V4 DeepSeek

DeepSeek-V4 (Flash o Pro) **non ha** un release GGUF. La conversione
HF→GGUF richiede che `llama.cpp` aggiunga supporto all'architettura
MLA+MoE+hyper-connections specifica di V4. Storicamente
`llama.cpp` impiega 2-4 mesi per nuove architetture MoE; al momento
di questo commit esiste un PR aperto per DeepSeek-V2 ma non per V4.

Detto altrimenti: oggi questo loader può leggere un Llama-3-8B-F16,
ma il `Transformer.forward` di `DeepSeekKit` è hard-coded per
l'architettura V4 e non saprebbe cosa fare con quei pesi.

### 3. Naming bridge GGUF ↔ nostro Assembly

GGUF usa convenzioni Llama-style (`blk.0.attn_q.weight`,
`blk.0.ffn_up.weight`) che non mappano 1:1 sui nomi MLA di V4
(`layers.0.attn.wq_a.weight`, `layers.0.attn.wq_b.weight`,
`layers.0.attn.wkv.weight`, …). Una traduzione richiederebbe sia un
mapping nome→nome che, in alcuni casi, una ricomposizione di tensor
(es. `wq_a + wq_b` di MLA collassati nel single `attn_q` di Llama —
matematicamente non equivalente senza ricalcolo).

Per ora `Assembly.swift` resta invariato. La presenza di
`GGUFFile.tensorNames` permette di esplorare i nomi disponibili in
un GGUF di test.

## Roadmap del completamento (in ordine di valore)

1. **Q8_0 dequant kernel** + Swift wrapper.
   Costo: ~150 LOC MSL + ~80 LOC Swift. Sblocca lettura completa di
   modelli Q8_0 (i meno aggressivi tra le quantizzazioni di
   `llama.cpp`).
2. **Q4_0 dequant kernel**.
   Costo: ~200 LOC MSL + ~80 LOC Swift. Sblocca i tantissimi Q4_0
   release pre-K-quants.
3. **Q4_K dequant kernel**.
   Costo: ~400 LOC MSL + ~100 LOC Swift. Sblocca i Q4_K_M / Q4_K_S che
   dominano TheBloke / bartowski.
4. **Q5_K / Q6_K dequant kernels** — analoghi a Q4_K.
5. **Llama-3 / Mistral / Qwen architettura nel Transformer**.
   Senza questo, leggere un GGUF è puramente diagnostico. Costo
   stimato: una full implementation a sé (~3-4 settimane di lavoro
   per la prima architettura, poi ~1 settimana per ciascuna delle
   varianti).
6. **Naming bridge** in `Assembly.swift` — utile solo se 5 è fatto.

I primi tre punti sono pure additivi (mai breaking) e possono andare
uno per volta in PR separate.

## Test

`Tests/DeepSeekKitTests/GGUFTests.swift` copre:

- Round-trip header su GGUF sintetico in-memory (2 KV pairs, 1 tensor
  F32 2×3).
- Rifiuto di magic invalido (`badMagic`).
- Rifiuto di version sconosciuta (`unsupportedVersion`).
- Block sizing dei principali quant types (Q8_0 = 32×34, Q4_0 = 32×18,
  Q4_K = 256×144).
- Classificazione pass-through vs dequant per i dtype noti.

Tutti i test sono pure-CPU (niente GPU richiesta) e finiscono in <10
ms — non richiedono fixture esterne.

## Limitazioni note

- **Solo lettura, niente scrittura.** Non c'è writer. Se serve produrre
  GGUF, usare `llama.cpp/convert_hf_to_gguf.py`.
- **Solo mmap.** A differenza di `SafeTensorsFile` non c'è path
  preload. Per la lettura strutturale del header non serve.
- **Allineamento ipotizzato little-endian.** Tutti i GGUF in
  produzione lo sono; il parser non gestisce big-endian.
- **Cap 256 MB sulla finestra di parse dei metadati.** Tensor data
  vive oltre, non è parsata.
