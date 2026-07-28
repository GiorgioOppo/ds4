[English](README.md) | **Italiano**

# Test del motore Laguna

Copertura senza device degli helper CPU del motore (dequantizzazione della
riga di embedding Q8_0 contro l'encoder condiviso, rifiuto dei limiti) più lo
smoke test opt-in su pesi reali: con `DS4_LAGUNA_GGUF` puntato al file
Q4_K_M ufficiale carica uno stack troncato, esegue due passi di decode e
verifica logits finiti e tracciamento della posizione. La parità numerica
contro il motore C di riferimento è il gate separato documentato in
`docs/PORTING-GAPS.it.md`.
