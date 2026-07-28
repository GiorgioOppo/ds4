[English](README.md) | **Italiano**

# Kernel Metal

Le sorgenti dei kernel Metal (`.metal`) sono la **fonte di verità** per
l'esecuzione su GPU. Il flusso di lavoro completo runtime/wrapper e la politica
di validazione numerica sono in
[`docs/BACKEND-METAL.md`](../docs/BACKEND-METAL.it.md).
Vengono incorporate nel binario tramite:

```sh
make embed-kernels
```

Quel comando esegue `scripts/embed_kernels.sh` e rigenera
`Sources/DS4Metal/Runtime/Generated/KernelSources.swift`, così il runtime non
richiede una directory di kernel su disco. Lo stesso percorso di sorgenti
incorporate funziona con SwiftPM, con il `.xcodeproj` generato e con una `.app`
distribuita.

## Struttura

I kernel sono raggruppati per architettura, una directory ciascuna:

- [`deepseek/`](deepseek/): i kernel DeepSeek V4 più le operazioni generiche
  condivise (norm, softmax, argsort, copy/cast, …) che il grafo DeepSeek
  pilota oggi.
- [`glm5.2/`](glm5.2/): i kernel GLM 5.2 (`glm-dsa`).
- [`laguna/`](laguna/): i kernel Laguna S 2.1 (`laguna`), portati dal branch
  di riferimento `laguna-s2.1`.

Metal compila comunque UNA sola libreria dalla concatenazione di tutti i file,
quindi i nomi dei kernel restano globalmente univoci tra le directory. La
chiave incorporata e la voce `MetalRuntime.kernelFiles` restano il basename del
file; il loader su disco (`MetalRuntime(metalDir:)`) cerca in
`MetalRuntime.kernelSubdirectories` e accetta anche la struttura piatta legacy.

File principali per peso a runtime:

- `deepseek/moe.metal`: kernel matvec MoE per tutti i formati di quantizzazione
  supportati.
- `deepseek/flash_attn.metal`: kernel di attenzione.
- `deepseek/dense.metal`: helper per le proiezioni dense.
- `deepseek/dsv4_misc.metal`, `dsv4_hc.metal`, `dsv4_kv.metal`, `dsv4_rope.metal`:
  helper specifici di DeepSeek-V4.
- `glm5.2/glm52.metal`: router GLM 5.2 e primitive compact-DSA.
- `laguna/laguna.metal`: norm/RoPE per-testa Laguna, store KV ad anello F16,
  attention GQA gated (decode e prefill), reduce flash-attention gated e una
  proiezione densa Q6_K.
- kernel di utilità per normalizzazione, softmax, argsort, operazioni unarie e
  la relativa colla.

## Flusso di lavoro

1. Modifica la sorgente `.metal`.
2. Esegui `make embed-kernels`.
3. Aggiorna o aggiungi il wrapper Swift in `Sources/DS4Metal/Kernels/` quando
   la firma del kernel cambia.
4. Mantieni l'ordine dei file sincronizzato con `MetalRuntime.kernelFiles`.
5. Esegui i test dei kernel e dei grafi descritti in
   [`Tests/METAL-TESTS.md`](../Tests/METAL-TESTS.it.md).

## Stato dell'audit dei kernel (revisionato il 2026-07-13)

L'audit originale di basso livello copriva tutti i 19 file di kernel e i loro
wrapper di dispatch Swift. Lo stato è stato ricontrollato rispetto all'albero
corrente il 2026-07-13. Gli elementi risolti sono conservati qui perché i
vecchi report non vengano scambiati per difetti aperti:

- **Risolto:** il caricamento della tabella condivisa iq2_xxs non genera più
  race né va fuori dai limiti con `nsg=4`.
- **Risolto:** `fp8_kv_quantize` ora protegge `in_nope` quando
  `n_nope % 64 != 0`.
- **Risolto:** `kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16` ora ha la barriera
  richiesta prima di riutilizzare `sb` alla k-iterazione successiva, e il suo
  dispatch Swift alloca i 16 KiB di memoria threadgroup richiesti.
- **Protetto:** la geometria del router è cablata a 256 esperti e scala 1.5;
  `StreamingDecoder` rifiuta le forme incompatibili prima del dispatch.

Le voci rimanenti sono aperte ma dormienti sulla geometria di produzione
supportata, salvo dove un punto dica esplicitamente il contrario. Ripeti
l'audit e aggiungi test prima di rendere attiva una combinazione oggi
irraggiungibile:

- **Aperto/dormiente:** `moe.metal`
  `kernel_mul_mv_{table,addr}_q4_K_sum6_f32`: un id esperto non valido in
  QUALSIASI slot esegue `return` (dst mai scritto, contributi degli altri slot
  persi) invece di `continue` come la variante q2_K; alla variante addr manca
  anche il controllo `addr == 0` che la q2_K possiede.
- **Aperto/dormiente:** implementazioni matvec di `dense.metal` (e mul_mv di
  `moe.metal`): l'ultimo threadgroup parziale legge tutte le nr0 righe di pesi
  senza protezione — letture OOB se outDim non è un multiplo di nr0(*nsg).
  Tutte le dimensioni di produzione sono multipli.
- **Aperto/dormiente:** `dense.metal` `switch (args.nr0)` senza default: nr0
  fuori da {2,4} è un no-op silenzioso (dst stantio).
- **Aperto/dormiente:** nel kernel vec di `flash_attn.metal` la
  zero-inizializzazione di `ss4` scrive 512 B per simdgroup in una regione da
  256 B (oggi la sovrapposizione è zeri-prima-della-barriera — benigna, ma
  qualsiasi cambio di layout la trasforma in corruzione). I kernel non-vec
  (non dispatchati da Swift) usano `ushort` per `iq1` → overflow oltre 65535
  righe.
- **Aperto/limitato:** il top-6 bitonico di `dsv4_misc.metal` non è stabile
  per indice sulle parità esatte di probabilità (riferimento C: vince l'indice
  più basso); il routing hash_mode esiste nel kernel ma non è mai pilotato dal
  motore Swift nonostante nHashLayer=3 nella tabella delle forme — verificare
  rispetto a ds4.c prima di abilitarlo.
- **Aperto/dormiente:** gli stride di token di hcExpand4/hcWeightedSum in
  `dsv4_hc.metal` (`nb_post1`, `nb_comb2`, `nb_w1`) sono corretti solo per
  nTokens == 1 quando è associato il buffer split impacchettato (righe da
  96 B) — ogni sito di chiamata di produzione passa 1. L'orientamento comb
  (nb_comb0=4/nb_comb1=16, cioè trasposto rispetto ai nomi dst/src del kernel)
  corrisponde alla documentazione lato Swift ma andrebbe verificato rispetto a
  ds4.c quando disponibile.
- **Aperto/rischio di produzione a contesto lungo:** `dsv4_rope.metal` calcola
  theta in f32 senza riduzione 2π e la libreria compila con fast-math —
  l'errore di fase cresce con pos (~2-3% sulle coppie più veloci a
  pos ≈ 300k). Degradazione graduale della qualità su contesti molto lunghi,
  nessun NaN. Una correzione (riduzione fmod / precise::sincos) cambia la
  numerica rispetto al riferimento C — richiede prima una valutazione di
  parità su dispositivo.
- **Aperto/dormiente:** `GraphContext` associa le attivazioni con `offset: 0`
  (byteOffset onorato solo per i pesi matmul): passare una vista con offset
  (rowView, slice di staging) a rmsNorm/add/swiglu leggerebbe silenziosamente
  i dati sbagliati.
- **Aperto/limitato dai chiamanti:** `GraphCompressor` scrive la riga
  compressa emessa in `cache[comp.count]` senza un controllo del limite
  maxComp (oggi i chiamanti limitano pos a maxKeys).
