# DeepSeekV4/Decode/State

Buffer temporanei riutilizzati lungo il forward per evitare allocazioni per token.

## File principali

- [`DecodeScratch.swift`](DecodeScratch.swift): tensor scratch per proiezioni,
  attention, router, MoE, output e indici; include viste speciali per i percorsi
  densi quantizzati.

## Flusso e dipendenze

Lo scratch parte dalla capacità richiesta dal contesto effettivamente usato e
cresce geometricamente quando viene superato un nuovo high-water mark. La
capacità massima configurata resta soltanto un limite: non viene impegnata alla
prima domanda. La sostituzione avviene dopo il drain dei command buffer e lo
scratch continua a essere condiviso in sequenza dalle operazioni del grafo.
Non rappresenta lo stato persistente della conversazione: quello appartiene
alla KV cache.

## Regole di modifica

Documentare per ogni buffer forma logica, byte effettivi, dtype e durata. Le
alias/view sono ammesse solo se gli intervalli di vita non si sovrappongono; una
nuova allocazione nel percorso per-token richiede una motivazione misurata.
