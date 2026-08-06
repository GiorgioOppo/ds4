[English](README.md) | **Italiano**

# DeepSeekV4/MTP e DSpark

Componenti per lo speculative decode con la testa MTP di DeepSeek. Il sidecar
ha l'accessory id interno `mtp`, ma è escluso dal catalogo GUI dei modelli
principali. Piano e stato in `docs/SELF-SPECULATIVE.md`, sezione "Fase M".

- `MTPSidecar.swift` — Fase M1: apertura del GGUF sidecar, classificazione dei
  tensori nei ruoli d'interfaccia (eh_proj, embed_tokens, enorm, hnorm,
  shared_head.*) e report di validazione contro le dimensioni del modello
  principale. Solo metadati: nessun buffer GPU, nessun effetto sul decode.
  Esposto nella demo via `DS4_MTP_GGUF` (percorso esplicito, oppure `=1` per
  cercare `*MTP*.gguf` accanto al modello).

La Fase M2 (caricamento residente + forward del draft) va cablata SOLO dopo
aver letto il report M1 sul sidecar reale: nomi, forme e quant dei tensori del
blocco transformer MTP determinano il wiring, e indovinarli produce garbage
silenzioso.

`DSparkSupportModel`, nello stesso file sorgente già incluso nel target, gestisce
il nuovo artefatto DSpark multistadio. Valida tutti gli 81 ruoli tensoriali,
metadati, classi di quantizzazione e forme contro la geometria Flash attiva. La
demo lo apre con `DS4_DSPARK_GGUF=<percorso>` oppure `=1`; la ricerca automatica
non mescola mai checkpoint 0730 e 0731. `DSparkStage0Runtime` cattura poi
la media degli stream HC in ogni target layer dichiarato dal supporto ed esegue
su Metal lo stage residente `main_proj` + `main_norm` dopo decode e prefill. I
tre transformer vengono già legati ai tipi ufficiali: matrici grandi ed esperti
restano viste mmap no-copy, mentre norm/scale/bias sono copiati residenti. Il
prefill conserva gli hidden target batchati e al primo uso li trasforma, a tile,
nei ring KV privati dei tre stadi. Il forward Metal esegue il blocco
`[target + 5 draft]` attraverso HC, proiezioni batch, attention non causale e
MoE routed IQ2_XXS/Q2_K.
`DSparkGreedyVerifier` e `dsparkVerifyAndCommit` implementano la verifica sul
target. Un'accettazione completa conserva il frontier del verificatore già
calcolato; una parziale fa rollback e replay esatto del prefisso accettato.
Il collasso HC finale, la testa Q8 condivisa, la correzione Markov sequenziale,
il gate confidence stabile e l'argmax sul device producono ora fino a cinque
token draft. Motore e demo inviano la proposta al verificatore e committano solo
il prefisso greedy esatto. DSpark si attiva intenzionalmente solo con temperatura
0 e repetition penalty 1; il sampling stocastico conserva il percorso ordinario.
`DS4_DSPARK_EXACT_REPLAY=1` forza il replay anche dopo accettazioni complete per
diagnosi di parità rigorosa, rinunciando però al guadagno speculativo.
