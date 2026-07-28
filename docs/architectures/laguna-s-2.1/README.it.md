[English](README.md) | **Italiano**

# Backend Laguna S 2.1 — frontend a stadi, decoder in sospeso

Laguna S 2.1 è il modello GQA + MoE di Poolside, supportato nativamente dal
motore C di riferimento sul branch `laguna-s2.1` di `antirez/ds4`. In questo
port la famiglia è **riconosciuta e completamente validata, ma non
eseguibile**: il frontend completo è implementato e coperto da unit test, e
l'inferenza è rifiutata dietro `LagunaRuntimeGate` finché il decoder Metal non
sarà portato.

## Geometria (esatta `DS4_SHAPE_LAGUNA_S21`)

48 blocchi · embedding 3072 · vocab 100352 · GQA con 8 teste KV, head-dim 128
· teste query per-layer: 48 ogni quarto blocco (full attention, 64 dimensioni
RoPE YaRN, base 500000, scala 32 su un contesto originale di 8192), 72 altrove
(sliding window di 512 token, 128 dimensioni RoPE, base 10000) · attention
gated con RMS-norm per-testa di Q/K · un blocco denso iniziale (FFN 12288) ·
256 esperti instradati, top-10, funzione di gating 2, scala 2.5, FFN esperti
1024, più un esperto condiviso · epsilon RMS 1e-6 · contesto distribuito
262144.

## Già in place (unit test, nessun file di modello richiesto)

- identificazione `laguna` e rilevamento famiglia con errore dedicato di
  non-implementato (`ModelArchitectureID.laguna`);
- validazione esatta di geometria/metadati inclusa l'alternanza 48/72 delle
  teste e il requisito YaRN (`LagunaConfiguration`);
- tokenizer BPE con il pre-split Laguna (prima le sequenze di LF, poi i
  gruppi in forma GLM4 a una cifra) e i token di controllo della famiglia
  (`LagunaTokenizer`);
- template chat nativo: apertura `〈|EOS|〉`, tag testuali `<system>`/`<user>`/
  `<tool_response>`, turni `<assistant>…</assistant>` con reasoning
  interlacciato `<think>` (`LagunaChatRenderer`);
- tool-call taggati (`<tool_call>nome<arg_key>…<arg_value>…`), parser strict
  e streaming, neutralizzazione del contenuto non fidato (`LagunaToolCodec`);
- default di sampling di riferimento: temperatura 0.7, top-k 20, top-p 0.95,
  min-p 0.05;
- schema tensori delle ricette pubblicate — signal path Q8_0, legacy
  Q4_K/F16, misto RoutedQ2_K/Last27Q3_K (`LagunaTensorSchema`);
- catalogo download: Q4_K_M ufficiale Poolside (con revision pinnata), il
  requant misto e il draft DFlash Q8_0 come accessorio.

## Non ancora implementato

- il decoder Metal (`metal/laguna.metal` + driver) e il suo gate di parità
  dei logits — il piano chiavi in mano è il Gap 4 in
  [`../../PORTING-GAPS.it.md`](../../PORTING-GAPS.it.md);
- il decoding speculativo DFlash (dopo il decoder);
- byte count/SHA-256 pinnati per gli artefatti a catalogo (richiesti prima
  che le voci possano diventare `runnable`);
- streaming SSD, inferenza distribuita e tensor parallelism (anche upstream
  richiede residenza completa per Laguna).
