[English](README.md) | **Italiano**

# Test dei riferimenti Laguna

Copertura senza device degli oracoli CPU Laguna con fixture calcolabili a
mano: alternanza delle spec di layer, RMS norm, guardia del gate softplus,
SwiGLU senza clamp, rotazione a coppie NeoX con e senza YaRN (incluso il
fattore di ampiezza a posizione 0), dimensioni di correzione YaRN, il router a
10 esperti (selezione con bias, pesi senza bias, ordine dei pareggi),
l'attention GQA gated sulla finestra ad anello F16 e le composizioni dei
blocchi attention/denso/MoE/testa di output.
