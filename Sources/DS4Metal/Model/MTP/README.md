# MTP (Multi-Token Prediction)

Componenti per lo speculative decode con la testa MTP di DeepSeek (il sidecar
`mtp` del catalogo download). Piano e stato in `docs/SELF-SPECULATIVE.md`,
sezione "Fase M".

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
