# A/B dei kernel Metal su M1 Pro — 16 luglio 2026

Questa è una fotografia sperimentale, non una promessa valida per ogni Apple
GPU. Le prove sono state eseguite su un MacBook Pro con M1 Pro e 16 GB, usando
il modello Flash locale da circa 91 GB, `DS4_RAW_RING=1`, context 4096, prompt da
133 token, decode greedy di 16 token e due token di warm-up. Expert bundle e
MetalIO erano disabilitati per evitare che una sidecar non corrispondente al
modello alterasse il confronto.

Ogni knob è stato provato in due processi separati e poi ripetuto invertendo
l'ordine. Per ciascuna coppia il runner ha confrontato token, argmax, top-3 e
2.197.760 logits Float32 con tolleranza zero.

| Candidato | Correttezza nei due ordini | Prefill, candidato vs base | Decode profile, candidato vs base | Media decode bilanciata | Decisione |
|---|---|---:|---:|---:|---|
| `DS4_FLASH_KV_STAGE=1` | `PASS_EXACT` | +3,5% / +1,4% | -1,8% / +2,3% | circa +0,2% | Opt-in: piccolo guadagno prefill, decode neutro |
| `DS4_VECTOR_COPY=1` | `PASS_EXACT` | 0,0% / +11,7% | -3,5% / 0,0% | circa -1,8% | Opt-in: prefill rumoroso, decode leggermente peggiore |
| `DS4_ROPE_PAIR=1` con affine | `PASS_EXACT` | -14,4% / -0,3% | -2,6% / +1,4% | circa -0,7% | Opt-in: nessun vantaggio end-to-end su M1 Pro |

La media bilanciata è la media aritmetica dei tok/s baseline e candidato nei due
ordini, seguita dal loro rapporto. Non elimina tutto il rumore, ma impedisce di
attribuire automaticamente al kernel il vantaggio del primo o del secondo
processo. Le anomalie di prefill in `VECTOR_COPY` e `ROPE_PAIR` mostrano perché
una singola coppia non è sufficiente per cambiare i default.

## Gate numerico e test GPU

Oltre al modello completo, i test GPU coprono:

- raw KV ring con wrap, cache compressa e mask NSA;
- blocco FlashAttention parziale e completo;
- conversioni F32/F16 con code scalari e trasporto di tutti i bit F16;
- RoPE baseline, pair e affine in decode, prefill, modalità inversa e YaRN.

Il gate locale riproducibile è:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift test --disable-sandbox \
  --filter 'GraphKVStageABTests|MetalCopyTests|MetalRoPETests'
```

Per ripetere un confronto full-model usare `scripts/metal_ab.sh`, prima con
l'ordine predefinito e poi invertito:

```sh
DS4_AB_ATOL=0 DS4_AB_RTOL=0 \
  scripts/metal_ab.sh model.gguf prompt.txt DS4_FLASH_KV_STAGE 0 1 16 out/base-first

DS4_AB_ATOL=0 DS4_AB_RTOL=0 DS4_AB_ORDER=candidate-first \
  scripts/metal_ab.sh model.gguf prompt.txt DS4_FLASH_KV_STAGE 0 1 16 out/candidate-first
```

I tre percorsi restano disponibili per misure su M3/M4 e contesti più lunghi,
ma non vengono applicati automaticamente dalla demo o dalla GUI finché un A/B
bilanciato sulla macchina target non mostra un vantaggio ripetibile.
